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

import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.LightHelpers;
import Scene.Raytracing;
import Scene.Intersection;
import Utils.Color.ColorHelpers;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;

enum class OutputMode : uint32_t
{
    Default = 0u,
    Luminance = 1u,
    ColorDistribution = 2u,
    Combined = 3u,
};
cbuffer CB
{
    uint gFrameCount;    // Frame count since scene was loaded.
    uint gPRNGDimension; // First available PRNG dimension.

    uint gIteration; // ReSTIR iteration
    OutputMode gOutputMode;
    uint gAnalyticalSamples;
    uint gEnvironmentSamples;
    uint gEmissiveSamples;
    uint gBsdfSamples;
    bool gCandidatesVisibility;
    uint gMaxConfidence;
    bool gTemporalReuse;
    uint gMaxSpatialSearch;
    uint gSpatialRadius;
}

uniform EnvMapSampler gEnvMapSampler;
uniform EmissiveLightSampler gEmissiveSampler;

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
struct EmissiveSample
{
    float2 barycentric;
    uint triangleIndex;
    float distSqr;
    float cosine; // dot product between normal of the triangle and the -light direction

    __init(float2 barycentric_ = float2(0, 0), uint triangleIndex_ = 0, float distSqr_ = 0.f, float cosine_ = 0.f)
    {
        barycentric = barycentric_;
        triangleIndex = triangleIndex_;
        distSqr = distSqr_;
        cosine = cosine_;
    }

    float3 getFullBarycentric() { return float3(1 - barycentric.x - barycentric.y, barycentric); }
}
struct ReSTIRSample
{
    int type;                // 0 = null, 1 = light, 2 = dir, 3 = emissive, 4 = half vector
    uint light;              // analytical
    float3 dir;              // environment
    EmissiveSample emissive; // emissive

    __init(int type_ = 0, uint light_ = 0, float3 dir_ = float3(0, 0, 0), EmissiveSample emissive_ = EmissiveSample())
    {
        type = type_;
        light = light_;
        dir = dir_;
        emissive = emissive_;
    }
}
float sumColor(float3 color)
{
    return color.r + color.g + color.b;
}
struct LightColor
{
    float3 color;

    __init(float3 light = float3(0.f))
    {
        float denom = sumColor(light);
        color = denom == 0 ? light : light / denom;
    }

    float3 apply(float light) { return light * color; }

    static LightColor combine(LightColor a, float wa, LightColor b, float wb) { return LightColor(a.color * wa + b.color * wb); }

    struct Builder
    {
        float3 color;

        __init(float3 color_ = float3(0.f)) { color = color_; }

        [mutating]
        void update(LightColor light, float weight) { color += light.apply(weight); }

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
        Phat phat;

        __init()
        {
            Y = ReSTIRSample();
            Wsum = 0.f;
            c = 0;
            phat = Phat();
        }

        /// Also returns the weight w_i
        [mutating]
        float update(Reservoir r, float m, Phat phat_, inout SampleGenerator sg)
        {
            float wi = m * phat_.phat * r.W * phat_.jacobian;
            Wsum += wi;
            c += r.c;
            if (sampleNext1D(sg) * Wsum < wi)
            {
                Y = r.Y;
                phat = phat_;
            }
            return wi;
        }

        Reservoir build()
        {
            float phatV = phat.phat;
            return Reservoir(Y, phatV == 0 ? 0 : Wsum / phatV, min(c, gMaxConfidence), phatV);
        }
    }

    __init(ReSTIRSample Y_ = ReSTIRSample(), float W_ = 0.f, int c_ = 0, float phat_ = 0.f)
    {
        Y = Y_;
        W = W_;
        c = c_;
        phat = phat_;
    }

    [mutating]
    void nullify()
    {
        Y = ReSTIRSample();
        W = 0;
        // c is kept
        phat = 0;
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
 * Payload for scatter ray (up to 72B).
 */
struct ScatterRayData
{
    float3 origin;
    float3 direction;
    PackedHitInfo hit;

    /**
     * Initializes ray payload with default parameters.
     */
    __init()
    {
        origin = float3(0);
        direction = float3(0);
        hit = {};
    }
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

uint getMaterialInstanceHints(const HitInfo hit)
{
    uint hints = 0;
    if (hit.getType() == HitType::Triangle || hit.getType() == HitType::DisplacedTriangle)
    {
        hints |= (uint)MaterialInstanceHints::AdjustShadingNormal;
    }
    return hints;
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
        mi = gScene.materials.getMaterialInstance(sd, lod, getMaterialInstanceHints(hit));

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
                if (isInBounds(neighbor, frameDim))
                {
                    if (abs(z - gLinearZ[neighbor].x) / (zGrad[dir] + 1e-2) < 2.f)
                    {
                        float3 otherN = gNormals[neighbor].xyz;
                        nGrad[dir] = min(nGrad[dir], distance(normal, otherN));
                    }
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

void sampleAnalyticalLight(inout SampleGenerator sg, out ReSTIRSample sample, out float invPdf)
{
    const uint lightCount = gScene.getLightCount();
    if (kUseAnalyticLights && lightCount > 0)
    {
        sample.type = 1;
        // Pick one of the analytic light sources randomly with equal probability.
        sample.light = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);
        invPdf = lightCount; // Light selection pdf = 1.0 / lightCount.
    }
    else
    {
        sample = {};
        invPdf = 1;
    }
}

void sampleEnvMap(inout SampleGenerator sg, out ReSTIRSample sample, out float invPdf)
{
    sample = {};
    invPdf = 1;
    if (kUseEnvLight)
    {
        EnvMapSample mapSample;
        if (gEnvMapSampler.sample(sampleNext2D(sg), mapSample))
        {
            if (mapSample.pdf > 0)
            {
                sample.type = 2;
                sample.dir = mapSample.dir;
                invPdf = 1.f / mapSample.pdf;
            }
        }
    }
}

void sampleEmissiveLight(ShadingData sd, inout SampleGenerator sg, out ReSTIRSample sample, out float invPdf)
{
    sample = {};
    invPdf = 1;
    if (kUseEmissiveLights)
    {
        TriangleLightSample tls;
        if (gEmissiveSampler.sampleLight(sd.posW, sd.getOrientedFaceNormal(), false, sg, tls))
        {
            if (tls.pdf > 0)
            {
                sample.type = 3;
                sample.emissive = EmissiveSample(
                    sample_triangle(tls.uv).yz, tls.triangleIndex, tls.distance * tls.distance, max(0, dot(-tls.dir, tls.normalW))
                );
                invPdf = 1.f / tls.pdf;
            }
        }
    }
}

void sampleBsdf(const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg, out ReSTIRSample sample, out float invPdf)
{
    sample = {};
    invPdf = 1;

    BSDFSample bsdfSample;
    if (mi.sample(sd, sg, bsdfSample, kUseImportanceSampling))
    {
        sample.type = 4;
        sample.dir = sd.frame.toLocal(normalizeSafe(sd.V + bsdfSample.wo));
        invPdf = 1.f / bsdfSample.pdf;
    }
}

struct Phat
{
    float3 dir;
    float distance;
    float3 full;
    float phat;
    float jacobian;

    __init()
    {
        dir = float3(0);
        distance = 0;
        full = float3(0);
        phat = 0;
        jacobian = 1;
    }
}
Phat evalF(
    const ShadingData sd,
    const IMaterialInstance mi,
    inout SampleGenerator sg,
    inout ReSTIRSample X,
    bool checkVisibility,
    bool updateX = false
)
{
    Phat phat = Phat();

    float3 Li;
    if (X.type == 1)
    {
        AnalyticLightSample ls;
        if (!sampleLight(sd.posW, gScene.getLight(X.light), sg, ls))
            return phat;
        phat.dir = ls.dir;
        Li = ls.Li;
        phat.distance = ls.distance;
    }
    else if (X.type == 2)
    {
        phat.dir = X.dir;
        Li = gEnvMapSampler.eval(phat.dir);
        phat.distance = kRayTMax;
    }
    else if (X.type == 3)
    {
        if (X.emissive.triangleIndex >= gScene.lightCollection.getTriangleCount())
            return phat;
        var triangle = gScene.lightCollection.getTriangle(X.emissive.triangleIndex);
        var barycentric = X.emissive.getFullBarycentric();
        var uv = triangle.getTexCoord(barycentric);
        var initialPos = triangle.getPosition(barycentric);
        {
            float distSqr = dot(initialPos - sd.posW, initialPos - sd.posW);
            float cosine = max(0, dot(-normalizeSafe(initialPos - sd.posW), triangle.normal));
            float denom = (X.emissive.cosine * distSqr);
            phat.jacobian = denom == 0 ? 0 : (cosine * X.emissive.distSqr) / denom;
            if (updateX)
            {
                X.emissive.distSqr = distSqr;
                X.emissive.cosine = cosine;
            }
        }
        Li = gScene.materials.evalEmissive(triangle.materialID, uv);

        // Use points with offsets to avoid self intersection
        var safePos = computeRayOrigin(initialPos, triangle.normal);
        var origin = computeRayOrigin(sd.posW, dot(sd.faceN, safePos - sd.posW) >= 0.f ? sd.faceN : -sd.faceN);
        phat.dir = normalizeSafe(safePos - origin);
        phat.distance = length(safePos - origin);
    }
    else if (X.type == 4)
    {
        checkVisibility = false;

        float3 halfVector = sd.frame.fromLocal(X.dir);
        ScatterRayData rayData = ScatterRayData();
        rayData.direction = reflect(-sd.V, halfVector);
        rayData.origin = sd.computeRayOrigin(dot(rayData.direction, sd.getOrientedFaceNormal()) >= 0);
        traceScatterRay(rayData);

        HitInfo hit = HitInfo(rayData.hit);
        if (hit.isValid())
        {
            bool shadowed = true;
            if (hit.getType() == HitType::Triangle)
            {
                TriangleHit triangleHit = hit.getTriangleHit();
                uint index = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
                if (index != gScene.lightCollection.kInvalidIndex)
                {
                    shadowed = false;

                    let lod = ExplicitLodTextureSampler(0.f);
                    VertexData v = gScene.getVertexData(triangleHit);
                    uint materialID = gScene.getMaterialID(triangleHit.instanceID);
                    ShadingData hitSD = gScene.materials.prepareShadingData(v, materialID, -rayData.direction, lod);
                    float distanceSqr = dot(hitSD.posW - sd.posW, hitSD.posW - sd.posW);

                    Li = gScene.materials.evalEmissive(materialID, v.texC);
                    phat.dir = rayData.direction;
                    phat.distance = sqrt(distanceSqr);
                }
            }
            if (shadowed)
            {
                return phat;
            }
        }
        else
        {
            phat.dir = rayData.direction;
            Li = gEnvMapSampler.eval(phat.dir);
            phat.distance = kRayTMax;
        }
    }
    else
    {
        return phat;
    }

    // Reject sample if not in the hemisphere of a BSDF lobe.
    const uint lobeTypes = mi.getLobeTypes(sd);
    const bool hasReflection = lobeTypes & uint(LobeType::Reflection);
    const bool hasTransmission = lobeTypes & uint(LobeType::Transmission);
    float NdotL = dot(sd.getOrientedFaceNormal(), phat.dir);
    if ((NdotL <= kMinCosTheta && !hasTransmission) || (NdotL >= -kMinCosTheta && !hasReflection))
        return phat;

    if (checkVisibility)
    {
        // Get origin with offset applied in direction of the geometry normal to avoid self-intersection.
        const float3 origin = computeRayOrigin(sd.posW, dot(sd.faceN, phat.dir) >= 0.f ? sd.faceN : -sd.faceN);

        // Test visibility by tracing a shadow ray.
        if (!traceShadowRay(origin, phat.dir, phat.distance))
            return phat;
    }

    // Evaluate contribution.
    phat.full = mi.eval(sd, phat.dir, sg) * Li;
    phat.phat = sumColor(phat.full);
    return phat;
}

Reservoir::Builder getCandidate(ReSTIRPoint point, inout SampleGenerator sg)
{
    // Generate the canonical ReSTIR samples
    var builder = Reservoir::Builder();
    {
        ReSTIRSample X;
        float invPdf;
        for (int i = 0; i < gAnalyticalSamples; ++i)
        {
            sampleAnalyticalLight(sg, X, invPdf);
            var phat = evalF(point.sd, point.mi, sg, X, gCandidatesVisibility);
            builder.update(Reservoir(X, invPdf, 1), 1.0f / gAnalyticalSamples, phat, sg);
        }
        for (int i = 0; i < gEnvironmentSamples; ++i)
        {
            sampleEnvMap(sg, X, invPdf);
            var phat = evalF(point.sd, point.mi, sg, X, gCandidatesVisibility);
            builder.update(Reservoir(X, invPdf, 1), 1.0f / gEnvironmentSamples, phat, sg);
        }
        for (int i = 0; i < gEmissiveSamples; ++i)
        {
            sampleEmissiveLight(point.sd, sg, X, invPdf);
            var phat = evalF(point.sd, point.mi, sg, X, gCandidatesVisibility);
            builder.update(Reservoir(X, invPdf, 1), 1.0f / gEmissiveSamples, phat, sg);
        }

        if (gBsdfSamples > 0)
        {
            var candidate = builder.build();
            var candidatePhat = builder.phat;
            builder = Reservoir::Builder();
            if (!gCandidatesVisibility)
            {
                // Reset sample
                candidatePhat = evalF(point.sd, point.mi, sg, candidate.Y, true);
                if (candidatePhat.phat == 0)
                {
                    candidate.nullify();
                }
            }
            float mCandidate = 1;
            if (candidate.Y.type == 2)
                mCandidate = (float)gEnvironmentSamples / (gEnvironmentSamples + gBsdfSamples);
            else if (candidate.Y.type == 3)
                mCandidate = (float)gEmissiveSamples / (gEmissiveSamples + gBsdfSamples);
            builder.update(candidate, mCandidate, candidatePhat, sg);
            for (int i = 0; i < gBsdfSamples; ++i)
            {
                sampleBsdf(point.sd, point.mi, sg, X, invPdf);
                var phat = evalF(point.sd, point.mi, sg, X, false);
                float mBSDF = 1;
                if (phat.distance == kRayTMax)
                    mBSDF = 1.f / (gEnvironmentSamples + gBsdfSamples);
                else
                    mBSDF = 1.f / (gEmissiveSamples + gBsdfSamples);
                builder.update(Reservoir(X, invPdf, 1), mBSDF, phat, sg);
            }
        }
        else if (!gCandidatesVisibility)
        {
            var candidate = builder.build();
            var candidatePhat = evalF(point.sd, point.mi, sg, candidate.Y, true);
            if (candidatePhat.phat == 0)
            {
                candidate.nullify();
            }
            builder = Reservoir::Builder();
            builder.update(candidate, 1, candidatePhat, sg);
        }
    }

    // Normalize confidence scale so that 1 is one frame of candidates
    builder.c = 1;
    return builder;
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
            Reservoir::Builder candidateBuilder = getCandidate(point, sg);
            var candidate = candidateBuilder.build();
            // Add visibility term if we don't already have it
            Phat phatCandidate = candidateBuilder.phat;
            LightColor candidateColor = LightColor(phatCandidate.full);

            uint2 oldPixel = reproject(pixel, frameDim);
            Temporal old = Temporal();
            if (gTemporalReuse && grads.allowTemporalReuse(pixel, oldPixel, frameDim, z, normal))
            {
                old = gPrevReSTIR[toIndex(oldPixel, frameDim)];
            }

            Phat phatOld = evalF(point.sd, point.mi, sg, old.r.Y, true, true);
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
                if (isInBounds(neighbor, frameDim))
                {
                    if (points[pointsCount].initialize(neighbor, frameDim))
                    {
                        if (grads.allowSpatialReuse(pixel, neighbor, z, normal))
                        {
                            points[pointsCount].r = gPrevReSTIR[toIndex(neighbor, frameDim)].r;
                            ++pointsCount;
                        }
                    }
                }
            }
            if (pointsCount > 1)
            {
                var builder = Reservoir::Builder();
                var colorBuilder = LightColor::Builder();
                for (int i = 0; i < pointsCount; ++i)
                {
                    Phat phat = evalF(points[0].sd, points[0].mi, sg, points[i].r.Y, true, true);
                    float mNumerator = 0.f;
                    float mDenominator = 0.f;
                    for (int j = 0; j < pointsCount; ++j)
                    {
                        Phat otherPhat = evalF(points[j].sd, points[j].mi, sg, points[i].r.Y, true);
                        var value = points[j].r.c * otherPhat.phat * otherPhat.jacobian;
                        mDenominator += value;
                        if (i == j)
                            mNumerator += value;
                    }
                    float m = mDenominator == 0 ? 0 : mNumerator / mDenominator;

                    float w = builder.update(points[i].r, m, phat, sg);
                    colorBuilder.update(i == 0 ? color : LightColor(phat.full), w);
                }
                // Don't increase confidence during spatial reuse
                builder.c = points[0].r.c;
                points[0].r = builder.build();
                color = colorBuilder.build();
            }
            gReSTIR[toIndex(pixel, frameDim)] = Temporal(points[0].r, color);

            {
                var props = point.mi.getProperties(point.sd);
                output.color = props.emission;
                output.albedo = props.diffuseReflectionAlbedo + props.diffuseTransmissionAlbedo + props.specularReflectionAlbedo +
                                props.specularTransmissionAlbedo;
            }
            switch (gOutputMode)
            {
            case OutputMode::Default:
                output.color += evalF(points[0].sd, points[0].mi, sg, points[0].r.Y, true).full * points[0].r.W;
                break;
            case OutputMode::ColorDistribution:
                output.color = color.color;
                break;
            case OutputMode::Luminance:
                output.color = evalF(points[0].sd, points[0].mi, sg, points[0].r.Y, true).phat * points[0].r.W;
                break;
            case OutputMode::Combined:
                output.color += color.apply(evalF(points[0].sd, points[0].mi, sg, points[0].r.Y, true).phat * points[0].r.W);
                break;
            default:
                output.color = float3(0.f);
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
    rayData.hit = {};
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
    rayData.hit = HitInfo(triangleHit).data;
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
    rayData.hit = HitInfo(displacedTriangleHit).data;
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
    rayData.hit = HitInfo(curveHit).data;
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
    rayData.hit = HitInfo(sdfGridHit).data;
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
