/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

/**
 * This implements ReSTIR for direct illumination, using the MinimalPathTracer example as a template.
 *
 * The host sets the following defines:
 *
 * SPLIT_CHANNELS          Whether or not the color channels should have their own reservoir.
 * MAX_SPATIAL_REUSE       Maximum number of spatial reuses.
 *
 * The following defines are from MinimalPathTracer
 *
 * MAX_BOUNCES             Maximum number of indirect bounces (0 means no indirect).
 * COMPUTE_DIRECT          Nonzero if direct illumination should be included.
 * USE_IMPORTANCE_SAMPLING Nonzero if importance sampling should be used for sampling materials.
 * USE_ANALYTIC_LIGHTS     Nonzero if Falcor's analytic lights should be used.
 * USE_EMISSIVE_LIGHTS     Nonzero if emissive geometry should be used as lights.
 * USE_ENV_LIGHT           Nonzero if env map is available and should be used as light source.
 * USE_ENV_BACKGROUND      Nonzero if env map is available and should be used as background.
 * is_valid_<name>         1 if optional I/O buffer with this name should be used.
 */

#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.LightHelpers;
import Scene.Raytracing;
import Scene.Intersection;
import Utils.Color.ColorHelpers;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;

cbuffer CB
{
    uint gFrameCount;    // Frame count since scene was loaded.
    uint gPRNGDimension; // First available PRNG dimension.

    uint gIteration; // ReSTIR iteration
    uint gCandidateCount;
    bool gCandidatesVisibility;
    bool gReuseCandidates;
    uint gMaxConfidence;
    bool gTemporalReuse;
    uint gMaxSpatialSearch;
    uint gSpatialRadius;
    bool gChannelReuse;
}

uniform EnvMapSampler gEnvMapSampler;

// Inputs
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float2> gMVec;
Texture2D<float4> gNormals;
Texture2D<float4> gPrevNormals;
Texture2D<float2> gLinearZ;
Texture2D<float2> gPrevLinearZ;
Texture2D<float4> gPosW;
Texture2D<float4> gPrevPosW;
Texture2D<float4> gViewW; // Optional (this must have a direction for all pixels if bound - GBufferRaster only does it for hits)

// Outputs
RWTexture2D<float4> gOutputColor;
RWTexture2D<float4> gOutputAlbedo;

// Internals
struct ReSTIRSample
{
    int type;   // 0 = null, 1 = light, 2 = dir
    uint light; // analytical
    float3 dir; // environment
                // todo: emissive

    __init(int type_ = 0, uint light_ = 0, float3 dir_ = float3(0, 0, 0))
    {
        type = type_;
        light = light_;
        dir = dir_;
    }
}
struct LightColor
{
    float3 color;

    __init(float3 light = float3(0.f))
    {
        float denom = light.r + light.g + light.b;
        color = denom == 0 ? light : light / denom;
    }

    float3 apply(float light) { return light * color; }

    static LightColor combine(LightColor a, float wa, LightColor b, float wb) { return LightColor(a.color * wa + b.color * wb); }

    struct Builder
    {
        float3 color;

        __init(float3 color_ = float3(0.f)) { color = color_; }

        Builder update(LightColor light, float weight) { return Builder(color + light.apply(weight)); }

        LightColor build() { return LightColor(color); }
    }
}
struct Reservoir
{
    ReSTIRSample Y;
    float W;
    int c;
    float phat;

    struct Builder
    {
        ReSTIRSample Y;
        float Wsum;
        int c;
        float phat;

        __init()
        {
            Y = ReSTIRSample();
            Wsum = 0.f;
            c = 0;
            phat = 0.f;
        }

        /// Also returns the weight w_i
        [mutating]
        float update(Reservoir r, float m, float phat_, inout SampleGenerator sg)
        {
            float wi = m * phat_ * r.W;
            Wsum += wi;
            c += r.c;
            if (sampleNext1D(sg) * Wsum < wi)
            {
                Y = r.Y;
                phat = phat_;
            }
            return wi;
        }

        Reservoir build() { return Reservoir(Y, phat == 0 ? 0 : Wsum / phat, min(c, gMaxConfidence), phat); }
    }

    __init(ReSTIRSample Y_ = ReSTIRSample(), float W_ = 0.f, int c_ = 0, float phat_ = 0.f)
    {
        Y = Y_;
        W = W_;
        c = c_;
        phat = phat_;
    }
}
struct Temporal
{
    Reservoir r;
    LightColor c;

    __init(Reservoir r_ = Reservoir(), LightColor c_ = LightColor())
    {
        r = r_;
        c = c_;
    }
}
RWStructuredBuffer<Temporal> gReSTIR;
StructuredBuffer<Temporal> gPrevReSTIR;

// Static configuration based on defines set from the host.
#define is_valid(name) (is_valid_##name != 0)
static const bool kSplitChannels = SPLIT_CHANNELS;
static const uint kSpatialReuse = SPATIAL_REUSE;

static const uint kMaxBounces = MAX_BOUNCES;
static const bool kComputeDirect = COMPUTE_DIRECT;
static const bool kUseImportanceSampling = USE_IMPORTANCE_SAMPLING;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const float3 kDefaultBackgroundColor = float3(0, 0, 0);
static const float kRayTMax = FLT_MAX;

/**
 * Payload for shadow ray.
 */
struct ShadowRayData
{
    bool visible;
};

/**
 * Setup ShadingData based on loaded vertex/material attributes for a hit point.
 * @param[in] hit Hit information.
 * @param[in] rayOrigin Ray origin.
 * @param[in] rayDir Normalized ray direction.
 * @param[in] lod Method for computing texture level-of-detail.
 * @return ShadingData struct.
 */
ShadingData loadShadingData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir, const ITextureSampler lod)
{
    VertexData v = {};
    uint materialID = {};

#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        v = gScene.getVertexData(triangleHit);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)
    if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
        v = gScene.getVertexData(displacedTriangleHit, -rayDir);
        materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE)
    if (hit.getType() == HitType::Curve)
    {
        const CurveHit curveHit = hit.getCurveHit();
        v = gScene.getVertexDataFromCurve(curveHit);
        materialID = gScene.getMaterialID(curveHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_SDF_GRID)
    if (hit.getType() == HitType::SDFGrid)
    {
        const SDFGridHit sdfGridHit = hit.getSDFGridHit();
        v = gScene.getVertexDataFromSDFGrid(sdfGridHit, rayOrigin, rayDir);
        materialID = gScene.getMaterialID(sdfGridHit.instanceID);
    }
#endif

    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);

    return sd;
}

/**
 * Returns the primary ray's direction.
 */
float3 getPrimaryRayDir(uint2 launchIndex, uint2 launchDim, const Camera camera)
{
    if (is_valid(gViewW))
    {
        // If we have the view vector bound as a buffer, just fetch it. No need to compute anything.
        return -gViewW[launchIndex].xyz;
    }
    else
    {
        // Compute the view vector. This must exactly match what the G-buffer pass is doing (jitter etc.).
        // Note that we do not take depth-of-field into account as it would require exactly matching the
        // sample generator between the passes, which is error prone. The host side will issue a warning instead.
        return camera.computeRayPinhole(launchIndex, launchDim).dir;
    }
}

/**
 * Traces a shadow ray towards a light source.
 * @param[in] origin Ray origin for the shadow ray.
 * @param[in] dir Direction from shading point towards the light source (normalized).
 * @param[in] distance Distance to the light source.
 * @return True if light is visible, false otherwise.
 */
bool traceShadowRay(float3 origin, float3 dir, float distance)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.f;
    ray.TMax = distance;

    ShadowRayData rayData;
    rayData.visible = false; // Set to true by miss shader if ray is not terminated before
    TraceRay(
        gScene.rtAccel,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xff /* instanceInclusionMask */,
        1 /* hitIdx */,
        rayTypeCount,
        1 /* missIdx */,
        ray,
        rayData
    );

    return rayData.visible;
}

bool isInBounds(const uint2 pixel, const uint2 frameDim)
{
    return all(pixel >= 0) && all(pixel < frameDim);
}

uint toIndex(const uint2 pixel, const uint2 frameDim)
{
    return pixel.x + pixel.y * frameDim.x;
}

float3 getForwardDirFromViewMat(float4x4 viewMat)
{
    return normalize(-viewMat[2].xyz);
}

struct ReSTIRPoint
{
    ShadingData sd;
    IMaterialInstance mi;
    Reservoir r;

    [mutating]
    bool initialize(uint2 pixel, uint2 frameDim)
    {
        if (!isInBounds(pixel, frameDim))
            return false;

        const HitInfo hit = HitInfo(gVBuffer[pixel]);
        if (!hit.isValid())
            return false;

        let lod = ExplicitLodTextureSampler(0.f);
        const float3 primaryRayOrigin = gScene.camera.getPosition();
        const float3 primaryRayDir = getPrimaryRayDir(pixel, frameDim, gScene.camera);

        // Load shading data.
        sd = loadShadingData(hit, primaryRayOrigin, primaryRayDir, lod);

        // Create material instance at shading point.
        mi = gScene.materials.getMaterialInstance(sd, lod);

        return true;
    }
}

int2 reproject(uint2 pixel, uint2 frameDim)
{
    // From SVGF reprojection
    // +0.5 to account for texel center offset
    return int2(float2(pixel) + gMVec[pixel].xy * frameDim + float2(0.5, 0.5));
}

struct GeometryGrads
{
    float2 zGrad;
    float2 nGrad;

    __init(uint2 pixel, uint2 frameDim, float3 pos, float z, float3 normal, float3 faceNormal)
    {
        // zGrad
        {
            uint2 neighbors[] = {
                pixel.x == 0 ? pixel + uint2(1, 0) : pixel - uint2(1, 0),
                pixel.y == 0 ? pixel + uint2(0, 1) : pixel - uint2(0, 1),
            };
            float3 front = normalize(gScene.camera.data.cameraW);
            for (int i = 0; i < 2; ++i)
            {
                // https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
                float3 l = getPrimaryRayDir(neighbors[i], frameDim, gScene.camera);
                float d = dot(pos - gScene.camera.data.posW, faceNormal) / dot(l, faceNormal);
                zGrad[i] = abs(z - dot(front, l * d));
            }
        }

        // nGrad
        {
            float defaultValue = 1e20;
            nGrad = float2(defaultValue);
            uint2 neighbors[4] = {
                pixel + uint2(1, 0),
                pixel + uint2(-1, 0),
                pixel + uint2(0, 1),
                pixel + uint2(0, -1),
            };
            for (int i = 0; i < 4; ++i)
            {
                uint dir = i / 2;
                uint2 neighbor = neighbors[i];
                if (isInBounds(neighbor, frameDim) && abs(z - gLinearZ[neighbor].x) / (zGrad[dir] + 1e-2) < 2.f)
                {
                    float3 otherN = gNormals[neighbor].xyz;
                    nGrad[dir] = min(nGrad[dir], distance(normal, otherN));
                }
            }

            if (nGrad.x == defaultValue)
                nGrad.x = 0.01;
            if (nGrad.y == defaultValue)
                nGrad.y = 0.01;
        }
    }

    bool allowSpatialReuse(uint2 pixel, uint2 neighbor, float z, float3 normal)
    {
        // From SVGF
        if (abs(gLinearZ[neighbor].x - z) / (max(zGrad.x, zGrad.y) + 1e-2f) > 10.f)
            return false;

        if (distance(gNormals[neighbor].xyz, normal) / (max(nGrad.x, nGrad.y) + 1e-2) > 16.0)
            return false;

        return true;
    }

    bool allowTemporalReuse(uint2 pixel, uint2 prevPixel, uint2 imageDim, float z, float3 normal)
    {
        // From SVGF
        if (!isInBounds(prevPixel, imageDim))
            return false;

        float3 dir = normalize(gScene.camera.data.cameraW);
        float movedZ = dot(gPrevPosW[prevPixel].xyz - gScene.camera.data.posW, dir);
        if (abs(movedZ - z) / (max(zGrad.x, zGrad.y) + 1e-2f) > 10.f)
            return false;

        if (distance(gPrevNormals[prevPixel].xyz, normal) / (max(nGrad.x, nGrad.y) + 1e-2) > 16.0)
            return false;

        return true;
    }
}

void sampleAnalyticalLight(inout SampleGenerator sg, out uint lightIndex, out float invPdf)
{
    // Assumes that lightCount > 0
    const uint lightCount = gScene.getLightCount();
    // Pick one of the analytic light sources randomly with equal probability.
    lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);
    invPdf = lightCount; // Light selection pdf = 1.0 / lightCount.
}

void sampleEnvMap(inout SampleGenerator sg, out float3 dir, out float invPdf)
{
    EnvMapSample sample;
    gEnvMapSampler.sample(sampleNext2D(sg), sample);
    dir = sample.dir;
    invPdf = 1.f / sample.pdf;
}

void sampleAllLights(inout SampleGenerator sg, out ReSTIRSample sample, out float invPdf)
{
    float prob = 0.5f;
    const uint lightCount = gScene.getLightCount();
    if (lightCount == 0)
        prob = 0.f;

    if (sampleNext1D(sg) < prob)
    {
        sample.type = 1;
        sampleAnalyticalLight(sg, sample.light, invPdf);
        invPdf /= prob;
    }
    else
    {
        sample.type = 2;
        sampleEnvMap(sg, sample.dir, invPdf);
        invPdf /= (1 - prob);
    }
}

float3 evalF(const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg, ReSTIRSample X, const bool checkVisibility)
{
    float3 dir;
    float3 Li;
    float distance;
    if (X.type == 1)
    {
        AnalyticLightSample ls;
        if (!sampleLight(sd.posW, gScene.getLight(X.light), sg, ls))
            return float3(0.f);
        dir = ls.dir;
        Li = ls.Li;
        distance = ls.distance;
    }
    else if (X.type == 2)
    {
        dir = X.dir;
        Li = kUseEnvBackground ? gScene.envMap.eval(dir) : kDefaultBackgroundColor;
        distance = kRayTMax;
    }
    else
    {
        return float3(0.f);
    }

    // Reject sample if not in the hemisphere of a BSDF lobe.
    const uint lobeTypes = mi.getLobeTypes(sd);
    const bool hasReflection = lobeTypes & uint(LobeType::Reflection);
    const bool hasTransmission = lobeTypes & uint(LobeType::Transmission);
    float NdotL = dot(sd.getOrientedFaceNormal(), dir);
    if ((NdotL <= kMinCosTheta && !hasTransmission) || (NdotL >= -kMinCosTheta && !hasReflection))
        return float3(0.f);

    if (checkVisibility)
    {
        // Get origin with offset applied in direction of the geometry normal to avoid self-intersection.
        const float3 origin = computeRayOrigin(sd.posW, dot(sd.faceN, dir) >= 0.f ? sd.faceN : -sd.faceN);

        // Test visibility by tracing a shadow ray.
        if (!traceShadowRay(origin, dir, distance))
            return float3(0.f);
    }

    // Evaluate contribution.
    return mi.eval(sd, dir, sg) * Li;
}
float evalPhat(const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg, ReSTIRSample X, const bool checkVisibility)
{
    float3 value = evalF(sd, mi, sg, X, checkVisibility);
    return value.r + value.g + value.b;
}

Reservoir getCandidate(ReSTIRPoint point, inout SampleGenerator sg)
{
    // Generate the canonical ReSTIR samples
    var candidateBuilder = Reservoir::Builder();
    for (int i = 0; i < gCandidateCount; ++i)
    {
        ReSTIRSample X;
        float invPdf;
        sampleAllLights(sg, X, invPdf);
        float phat = evalPhat(point.sd, point.mi, sg, X, gCandidatesVisibility);
        candidateBuilder.update(Reservoir(X, invPdf, 1, phat), 1.0f / gCandidateCount, phat, sg);
    }
    Reservoir candidate = candidateBuilder.build();
    // Normalize confidence scale so that 1 is one frame of candidates
    candidate.c = 1;
    return candidate;
}

struct Output
{
    float3 color;
    float3 albedo;
}
Output restirDirect(const uint2 pixel, const uint2 frameDim)
{
    Output output = Output(float3(0.f), float3(0.f));

    ReSTIRPoint point;
    if (point.initialize(pixel, frameDim))
    {
        // Create sample generator.
        // Makes sure that each iteration gets a different seed.
        SampleGenerator sg = SampleGenerator(pixel, (gFrameCount + 100) * (gIteration + 1));

        // Advance the generator to the first available dimension.
        // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
        for (uint i = 0; i < gPRNGDimension; i++)
            sampleNext1D(sg);

        // For geometry tests
        float z = gLinearZ[pixel].x;
        float3 normal = gNormals[pixel].xyz;
        GeometryGrads grads = GeometryGrads(pixel, frameDim, point.sd.posW, z, normal, point.sd.faceN);

        // ReSTIR
        if (gIteration == 0) // Candidate + Temporal
        {
            Reservoir candidate = getCandidate(point, sg);
            // Add visibility term if we don't already have it
            float phatCandidate = gCandidatesVisibility ? candidate.phat : evalPhat(point.sd, point.mi, sg, candidate.Y, true);
            LightColor candidateColor = LightColor(evalF(point.sd, point.mi, sg, candidate.Y, true));

            uint2 oldPixel = reproject(pixel, frameDim);
            Temporal old = Temporal();
            if (gTemporalReuse && grads.allowTemporalReuse(pixel, oldPixel, frameDim, z, normal))
            {
                old = gPrevReSTIR[toIndex(oldPixel, frameDim)];
            }

            float phatOld = evalPhat(point.sd, point.mi, sg, old.r.Y, true);
            var builder = Reservoir::Builder();
            float wc = builder.update(candidate, (float)candidate.c / (candidate.c + old.r.c), phatCandidate, sg);
            float wo = builder.update(old.r, (float)old.r.c / (candidate.c + old.r.c), phatOld, sg);
            point.r = builder.build();
            gReSTIR[toIndex(pixel, frameDim)] = Temporal(point.r, LightColor::combine(candidateColor, wc, old.c, wo));
        }
        else // Spatial + Channel
        {
            // Spatial
            ReSTIRPoint points[kSpatialReuse + 1];
            points[0] = point;
            points[0].r = gPrevReSTIR[toIndex(pixel, frameDim)].r;
            LightColor color = gPrevReSTIR[toIndex(pixel, frameDim)].c;
            int pointsCount = 1;
            for (int i = 0; i < gMaxSpatialSearch && pointsCount <= kSpatialReuse; ++i)
            {
                float2 offset = gSpatialRadius * (sampleNext2D(sg) * 2 - 1);
                uint2 neighbor = pixel + int2(offset + sign(offset));
                if (isInBounds(neighbor, frameDim) && points[pointsCount].initialize(neighbor, frameDim))
                {
                    if (grads.allowSpatialReuse(pixel, neighbor, z, normal))
                    {
                        points[pointsCount].r = gPrevReSTIR[toIndex(neighbor, frameDim)].r;
                        ++pointsCount;
                    }
                }
            }
            if (pointsCount > 1)
            {
                var builder = Reservoir::Builder();
                float ws[2];
                for (int i = 0; i < pointsCount; ++i)
                {
                    float phat = evalPhat(points[0].sd, points[0].mi, sg, points[i].r.Y, true);
                    float mNumerator = 0.f;
                    float mDenominator = 0.f;
                    for (int j = 0; j < pointsCount; ++j)
                    {
                        var value = points[j].r.c * evalPhat(points[j].sd, points[j].mi, sg, points[i].r.Y, true);
                        mDenominator += value;
                        if (i == j)
                            mNumerator += value;
                    }
                    float m = mDenominator == 0 ? 0 : mNumerator / mDenominator;

                    ws[i] = builder.update(points[i].r, m, phat, sg);
                }
                // Don't increase confidence during spatial reuse
                builder.c = points[0].r.c;
                points[0].r = builder.build();
                color = LightColor::combine(color, ws[0], LightColor(evalF(points[0].sd, points[0].mi, sg, points[1].r.Y, true)), ws[1]);
                // color = LightColor::combine(color, ws[0], LightColor(evalF(points[0].sd, points[0].mi, sg, points[1].r.Y, true)), ws[1] /
                // max(1, points[1].r.c));
            }
            gReSTIR[toIndex(pixel, frameDim)] = Temporal(points[0].r, color);

            {
                var props = point.mi.getProperties(point.sd);
                output.color = props.emission;
                output.albedo = props.diffuseReflectionAlbedo + props.diffuseTransmissionAlbedo + props.specularReflectionAlbedo +
                                props.specularTransmissionAlbedo;
            }
            if (kSplitChannels)
            {
                if (gChannelReuse)
                {
                    output.color += color.apply(evalPhat(points[0].sd, points[0].mi, sg, points[0].r.Y, true) * points[0].r.W);
                }
                else
                {
                    output.color = color.color;
                }
            }
            else
            {
                output.color += evalF(points[0].sd, points[0].mi, sg, points[0].r.Y, true) * points[0].r.W;
            }
        }
    }
    else
    {
        gReSTIR[toIndex(pixel, frameDim)] = Temporal();
        // Background pixel.
        const float3 primaryRayDir = getPrimaryRayDir(pixel, frameDim, gScene.camera);
        output.color = kUseEnvBackground ? gScene.envMap.eval(primaryRayDir) : kDefaultBackgroundColor;
        output.albedo = float3(1.f, 1.f, 1.f);
    }

    return output;
}

//
// Shader entry points for miss shaders.
//

[shader("miss")]
void scatterMiss(inout ScatterRayData rayData)
{
    // Ray missed the scene. Mark the ray as terminated.
    rayData.terminated = true;

    // Add contribution from distant light (env map) in this direction.
    if (kUseEnvLight && (kComputeDirect || rayData.pathLength > 0))
    {
        float3 Le = gScene.envMap.eval(WorldRayDirection());
        rayData.radiance += rayData.thp * Le;
    }
}

[shader("miss")]
void shadowMiss(inout ShadowRayData rayData)
{
    // The miss shader is executed if the ray misses all geometry. Mark as visible.
    rayData.visible = true;
}

//
// Shader entry points for TriangleMesh hit groups.
//

[shader("anyhit")]
void scatterTriangleMeshAnyHit(inout ScatterRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}

[shader("closesthit")]
void scatterTriangleMeshClosestHit(inout ScatterRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    handleHit(HitInfo(triangleHit), rayData);
}

[shader("anyhit")]
void shadowTriangleMeshAnyHit(inout ShadowRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}

//
// Shader entry points for DisplacedTriangleMesh hit groups.
//

[shader("intersection")]
void displacedTriangleMeshIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    DisplacedTriangleMeshIntersector::Attribs attribs;
    float t;
    if (DisplacedTriangleMeshIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void scatterDisplacedTriangleMeshClosestHit(inout ScatterRayData rayData, DisplacedTriangleMeshIntersector::Attribs attribs)
{
    DisplacedTriangleHit displacedTriangleHit;
    displacedTriangleHit.instanceID = getGeometryInstanceID();
    displacedTriangleHit.primitiveIndex = PrimitiveIndex();
    displacedTriangleHit.barycentrics = attribs.barycentrics;
    displacedTriangleHit.displacement = attribs.displacement;
    handleHit(HitInfo(displacedTriangleHit), rayData);
}

//
// Shader entry points for Curve hit groups.
//

[shader("intersection")]
void curveIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    CurveIntersector::Attribs attribs;
    float t;
    if (CurveIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void scatterCurveClosestHit(inout ScatterRayData rayData, CurveIntersector::Attribs attribs)
{
    CurveHit curveHit;
    curveHit.instanceID = getGeometryInstanceID();
    curveHit.primitiveIndex = PrimitiveIndex();
    curveHit.barycentrics = attribs.barycentrics;
    handleHit(HitInfo(curveHit), rayData);
}

//
// Shader entry points for SDFGrid hit groups.
//

[shader("intersection")]
void sdfGridIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    SDFGridHitData sdfGridHitData;
    float t;
    if (SDFGridIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), sdfGridHitData, t))
    {
        ReportHit(t, 0, sdfGridHitData);
    }
}

[shader("closesthit")]
void scatterSdfGridClosestHit(inout ScatterRayData rayData, SDFGridHitData sdfGridHitData)
{
    SDFGridHit sdfGridHit;
    sdfGridHit.instanceID = getGeometryInstanceID();
    sdfGridHit.hitData = sdfGridHitData;
    handleHit(HitInfo(sdfGridHit), rayData);
}

//
// Shader entry point for ray generation shader.
//

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;

    Output output = restirDirect(pixel, frameDim);

    gOutputColor[pixel] = float4(output.color, 1.f);
    gOutputAlbedo[pixel] = float4(output.albedo, 1.f);
}

//
// Unused functions from MinimalPathTracer
//

/**
 * Payload for scatter ray (up to 72B).
 */
struct ScatterRayData
{
    float3 radiance;  ///< Accumulated outgoing radiance from path.
    bool terminated;  ///< Set to true when path is terminated.
    float3 thp;       ///< Current path throughput. This is updated at each path vertex.
    uint pathLength;  ///< Path length in number of path segments (0 at origin, 1 at first secondary hit, etc.). Max 2^31.
    float3 origin;    ///< Next path segment origin.
    float3 direction; ///< Next path segment direction.

    SampleGenerator sg; ///< Per-ray state for the sample generator (up to 16B).

    /**
     * Initializes ray payload with default parameters.
     */
    __init(SampleGenerator sg)
    {
        this.terminated = false;
        this.pathLength = 0;
        this.radiance = float3(0, 0, 0);
        this.thp = float3(1, 1, 1);
        this.origin = float3(0, 0, 0);
        this.direction = float3(0, 0, 0);
        this.sg = sg;
    }
};

/**
 * Traces a scatter ray based on ray parameters stored in the ray payload.
 * @param[in] rayData Describes the ray parameters. The struct is modified based on the result.
 */
void traceScatterRay(inout ScatterRayData rayData)
{
    RayDesc ray;
    ray.Origin = rayData.origin;
    ray.Direction = rayData.direction;
    ray.TMin = 0.f;
    ray.TMax = kRayTMax;

    uint rayFlags = 0; // TODO: Set cull mode from the app
    TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayData);
}

/**
 * Evaluates the direct illumination from analytic lights.
 * This function samples Falcor's light list uniformly with one shadow ray.
 * @param[in] sd Shading data.
 * @param[in] mi Material instance.
 * @param[in,out] sg SampleGenerator object.
 * @return Outgoing radiance in view direction.
 */
float3 evalDirectAnalytic(const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg)
{
    const uint lightCount = gScene.getLightCount();
    if (lightCount == 0)
        return float3(0.f);

    // Pick one of the analytic light sources randomly with equal probability.
    const uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);
    float invPdf = lightCount; // Light selection pdf = 1.0 / lightCount.

    // Sample local light source.
    AnalyticLightSample ls;
    if (!sampleLight(sd.posW, gScene.getLight(lightIndex), sg, ls))
        return float3(0.f);

    // Reject sample if not in the hemisphere of a BSDF lobe.
    const uint lobeTypes = mi.getLobeTypes(sd);
    const bool hasReflection = lobeTypes & uint(LobeType::Reflection);
    const bool hasTransmission = lobeTypes & uint(LobeType::Transmission);
    float NdotL = dot(sd.getOrientedFaceNormal(), ls.dir);
    if ((NdotL <= kMinCosTheta && !hasTransmission) || (NdotL >= -kMinCosTheta && !hasReflection))
        return float3(0.f);

    // Get origin with offset applied in direction of the geometry normal to avoid self-intersection.
    const float3 origin = computeRayOrigin(sd.posW, dot(sd.faceN, ls.dir) >= 0.f ? sd.faceN : -sd.faceN);

    // Test visibility by tracing a shadow ray.
    bool V = traceShadowRay(origin, ls.dir, ls.distance);
    if (!V)
        return float3(0.f);

    // Evaluate contribution.
    return mi.eval(sd, ls.dir, sg) * ls.Li * invPdf;
}

/**
 * Generate a new scatter ray or terminate.
 * @param[in] sd Shading data.
 * @param[in] mi Material instance.
 * @param[in] isCurveHit True if on curve hit.
 * @param[in] rayOrigin Ray origin for the new ray.
 * @param[in,out] rayData Ray payload.
 * @return True if the path continues.
 */
bool generateScatterRay(const ShadingData sd, const IMaterialInstance mi, bool isCurveHit, float3 rayOrigin, inout ScatterRayData rayData)
{
    // Sample material.
    BSDFSample bsdfSample;
    if (mi.sample(sd, rayData.sg, bsdfSample, kUseImportanceSampling))
    {
        rayData.origin = rayOrigin;
        if (!isCurveHit && bsdfSample.isLobe(LobeType::Transmission))
        {
            rayData.origin = sd.computeRayOrigin(false);
        }
        rayData.direction = bsdfSample.wo;
        rayData.thp *= bsdfSample.weight;
        return any(rayData.thp > 0.f);
    }

    return false;
}

/**
 * Process a hit.
 * Loads the shading data, samples analytic lights and samples a new scatter ray.
 * Terminates the path if maximum number of bounces is reached.
 * @param[in] hit Hit info.
 * @param[in,out] rayData Ray payload.
 *
 */
void handleHit(const HitInfo hit, inout ScatterRayData rayData)
{
    const bool isCurveHit = hit.getType() == HitType::Curve;
    let lod = ExplicitLodTextureSampler(0.f);

    // Load shading data.
    ShadingData sd = loadShadingData(hit, rayData.origin, rayData.direction, lod);

    // Create material instance.
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    // Add emitted light.
    if (kUseEmissiveLights && (kComputeDirect || rayData.pathLength > 0))
    {
        rayData.radiance += rayData.thp * mi.getProperties(sd).emission;
    }

    // Check whether to terminate based on max depth.
    if (rayData.pathLength >= kMaxBounces)
    {
        rayData.terminated = true;
        return;
    }

    // Compute ray origin for new rays spawned from the hit.
    float3 rayOrigin;
    if (isCurveHit)
    {
        // For curves, we set the new origin at the sphere center.
        rayOrigin = sd.posW - sd.curveRadius * sd.frame.N;
    }
    else
    {
        rayOrigin = sd.computeRayOrigin();
    }

    // Add contribution of direct light from analytic lights.
    if (kUseAnalyticLights)
    {
        float3 Lr = evalDirectAnalytic(sd, mi, rayData.sg);
        rayData.radiance += rayData.thp * Lr;
    }

    // Generate scatter ray for the next path segment.
    // The raygen shader will continue the path based on the returned payload.
    if (!generateScatterRay(sd, mi, isCurveHit, rayOrigin, rayData))
    {
        rayData.terminated = true;
        return;
    }

    rayData.pathLength++;
}

/**
 * This is the main entry point for the minimal path tracer.
 *
 * One path per pixel is generated, which is traced into the scene.
 * The path tracer is written as a for-loop over path segments.
 *
 * Built-in light sources (point, directional) are sampled explicitly at each
 * path vertex. The contributions from area lights (env map and mesh lights)
 * are explicitly added by the scatter ray hit/miss shaders.
 *
 * @param[in] pixel Pixel to trace a path for.
 * @param[in] frameDim Dimension of the frame in pixels.
 * @return Returns the estimated color (radiance).
 */
float3 tracePath(const uint2 pixel, const uint2 frameDim)
{
    float3 outColor = float3(0.f);

    const float3 primaryRayOrigin = gScene.camera.getPosition();
    const float3 primaryRayDir = getPrimaryRayDir(pixel, frameDim, gScene.camera);

    const HitInfo hit = HitInfo(gVBuffer[pixel]);
    if (hit.isValid())
    {
        // Pixel represents a valid primary hit. Compute its contribution.

        const bool isCurveHit = hit.getType() == HitType::Curve;
        let lod = ExplicitLodTextureSampler(0.f);

        // Load shading data.
        ShadingData sd = loadShadingData(hit, primaryRayOrigin, primaryRayDir, lod);

        // Create material instance at shading point.
        let mi = gScene.materials.getMaterialInstance(sd, lod);

        // Create sample generator.
        SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

        // Advance the generator to the first available dimension.
        // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
        for (uint i = 0; i < gPRNGDimension; i++)
            sampleNext1D(sg);

        // Compute ray origin for new rays spawned from the G-buffer.
        float3 rayOrigin;
        if (isCurveHit)
        {
            // For curves, we set the new origin at the sphere center.
            rayOrigin = sd.posW - sd.curveRadius * sd.frame.N;
        }
        else
        {
            rayOrigin = sd.computeRayOrigin();
        }

        if (kComputeDirect)
        {
            // Always output directly emitted light, independent of whether emissive materials are treated as light sources or not.
            outColor += mi.getProperties(sd).emission;

            // Add contribution of direct light from analytic lights.
            // Light probe and mesh lights are handled by the scatter ray hit/miss shaders.
            outColor += kUseAnalyticLights ? evalDirectAnalytic(sd, mi, sg) : float3(0.f);
        }

        // Prepare ray payload.
        ScatterRayData rayData = ScatterRayData(sg);

        // Generate scatter ray.
        if (!generateScatterRay(sd, mi, isCurveHit, rayOrigin, rayData))
            rayData.terminated = true;

        // Follow path into the scene and compute its total contribution.
        for (uint depth = 0; depth <= kMaxBounces && !rayData.terminated; depth++)
        {
            // Trace scatter ray. If it hits geometry, the closest hit shader samples
            // direct illumination and generates the next scatter ray.
            traceScatterRay(rayData);
        }

        // Store contribution from scatter ray.
        outColor += rayData.radiance;
    }
    else
    {
        // Background pixel.
        outColor = kUseEnvBackground ? gScene.envMap.eval(primaryRayDir) : kDefaultBackgroundColor;
    }

    return outColor;
}
