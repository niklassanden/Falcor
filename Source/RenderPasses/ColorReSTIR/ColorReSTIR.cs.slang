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
import Scene.RaytracingInline;
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
}
enum class TemporalColorEstimate : uint32_t
{
    None = 0u,
    Full = 1u,
    Gradient = 2u,
}
cbuffer CB
{
    uint2 gFrameDim;
    uint gFrameCount;    // Frame count since scene was loaded.
    uint gPRNGDimension; // First available PRNG dimension.

    uint gIteration; // ReSTIR iteration
    OutputMode gOutputMode;
    TemporalColorEstimate gTemporalColorEstimate;
    bool gNormalizeColorEstimate;
    bool gReuseDemodulated;
    uint gAnalyticalSamples;
    uint gEnvironmentSamples;
    uint gEmissiveSamples;
    uint gDeltaSamples;
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
RWTexture2D<float4> gOutputDelta;
RWTexture2D<float4> gOutputAlbedo;

// Internals
struct EmissiveSample
{
    float2 barycentric;
    uint triangleIndex;

    __init(float2 barycentric_ = float2(0, 0), uint triangleIndex_ = 0, float distSqr_ = 0.f, float cosine_ = 0.f)
    {
        barycentric = barycentric_;
        triangleIndex = triangleIndex_;
    }

    float3 getFullBarycentric() { return float3(1 - barycentric.x - barycentric.y, barycentric); }
}
struct ReSTIRSample
{
    int type;                // 0 = null, 1 = light, 2 = dir, 3 = emissive
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

    __init(float3 color_ = float3(0.f)) { color = color_; }

    float3 apply(float light) { return light * color; }

    LightColor normalize(bool normalize = true)
    {
        if (normalize)
        {
            float denom = sumColor(color);
            return LightColor(denom == 0 ? color : color / denom);
        }
        return this;
    }

    static LightColor combine(LightColor a, float wa, LightColor b, float wb) { return LightColor(a.color * wa + b.color * wb); }

    struct Builder
    {
        float3 color;

        __init(float3 color_ = float3(0.f)) { color = color_; }

        [mutating]
        void update(LightColor color_, float weight) { color += color_.apply(weight); }

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
            float wi = m * phat_.phat * r.W;
            Wsum += wi;
            c += r.c;
            if (sampleNext1D(sg) * Wsum < wi)
            {
                Y = r.Y;
                phat = phat_;
            }
            return wi;
        }

        Reservoir build() { return Reservoir(Y, phat.phat == 0 ? 0 : Wsum / phat.phat, min(c, gMaxConfidence), phat.phat); }
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
    LightColor old;

    __init(Reservoir r_ = Reservoir(), LightColor c_ = LightColor(), LightColor old_ = LightColor())
    {
        r = r_;
        c = c_;
        old = old_;
    }
}
RWStructuredBuffer<Temporal> gReSTIR;
StructuredBuffer<Temporal> gPrevReSTIR;

// Static configuration based on defines set from the host.
#define is_valid(name) (is_valid_##name != 0)
static const uint kSpatialReuse = SPATIAL_REUSE;

static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const float3 kDefaultBackgroundColor = float3(0, 0, 0);
static const float kRayTMax = FLT_MAX;

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
float3 getPrimaryRayDir(uint2 launchIndex, const Camera camera)
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
        return camera.computeRayPinhole(launchIndex, gFrameDim).dir;
    }
}

bool isInBounds(const uint2 pixel)
{
    return all(pixel >= 0) && all(pixel < gFrameDim);
}

uint toIndex(const uint2 pixel)
{
    return pixel.x + pixel.y * gFrameDim.x;
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
    LightColor c;

    [mutating]
    bool initialize(uint2 pixel)
    {
        if (!isInBounds(pixel))
            return false;

        const HitInfo hit = HitInfo(gVBuffer[pixel]);
        if (!hit.isValid())
            return false;

        let lod = ExplicitLodTextureSampler(0.f);
        const float3 primaryRayOrigin = gScene.camera.getPosition();
        const float3 primaryRayDir = getPrimaryRayDir(pixel, gScene.camera);

        // Load shading data.
        sd = loadShadingData(hit, primaryRayOrigin, primaryRayDir, lod);
        // Handle delta distributions separately
        sd.mtl.setActiveLobes((uint)LobeType::NonDelta);

        // Create material instance at shading point.
        mi = gScene.materials.getMaterialInstance(sd, lod, getMaterialInstanceHints(hit));

        return true;
    }
}

int2 reproject(uint2 pixel)
{
    // From SVGF reprojection
    // +0.5 to account for texel center offset
    return int2(float2(pixel) + gMVec[pixel].xy * gFrameDim + float2(0.5, 0.5));
}

struct GeometryGrads
{
    float2 zGrad;
    float2 nGrad;

    __init(uint2 pixel, float3 pos, float z, float3 normal, float3 faceNormal)
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
                float3 l = getPrimaryRayDir(neighbors[i], gScene.camera);
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
                if (isInBounds(neighbor))
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

    bool allowTemporalReuse(uint2 pixel, uint2 prevPixel, float z, float3 normal)
    {
        // From SVGF
        if (!isInBounds(prevPixel))
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

#if USE_ANALYTIC_LIGHTS == 1
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
#endif

#if USE_ENV_LIGHT == 1
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
#endif

#if USE_EMISSIVE_LIGHTS == 1
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
                sample.emissive = EmissiveSample(sample_triangle(tls.uv).yz, tls.triangleIndex);
                invPdf = 1.f / tls.pdf * tls.distance * tls.distance / max(0, dot(-tls.dir, tls.normalW));
            }
        }
    }
}
#endif

float3 evalDelta(ShadingData sd, IMaterialInstance mi, inout SampleGenerator sg)
{
    sd.mtl.setActiveLobes((uint)LobeType::Delta);

    BSDFSample bsdfSample;
    if (mi.sample(sd, sg, bsdfSample, true))
    {
        const Ray ray = Ray(sd.computeRayOrigin(!bsdfSample.isLobe(LobeType::Transmission)), bsdfSample.wo, 0.f, kRayTMax);
        HitInfo hit;
        float hitT;
        if (traceSceneRay<1>(ray, hit, hitT, RAY_FLAG_NONE, 0xff) && hit.isValid())
        {
#if USE_EMISSIVE_LIGHTS == 1
            if (hit.getType() == HitType::Triangle)
            {
                TriangleHit triangleHit = hit.getTriangleHit();
                uint index = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
                if (index != gScene.lightCollection.kInvalidIndex)
                {
                    let lod = ExplicitLodTextureSampler(0.f);
                    VertexData v = gScene.getVertexData(triangleHit);
                    uint materialID = gScene.getMaterialID(triangleHit.instanceID);

                    return bsdfSample.weight * gScene.materials.evalEmissive(materialID, v.texC);
                }
            }
#endif
        }
        else
        {
#if USE_ENV_LIGHT == 1
            return bsdfSample.weight * gEnvMapSampler.eval(ray.dir);
#endif
        }
    }
    return float3(0);
}

struct Phat
{
    float3 full;
    float phat;

    __init()
    {
        full = float3(0);
        phat = 0;
    }
}
Phat evalF(const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg, inout ReSTIRSample X, bool checkVisibility)
{
    Phat phat = Phat();

    float3 dir = float3(0);
    float distance = 0;
    float3 Li;
    if (X.type == 1)
    {
#if USE_ANALYTIC_LIGHTS == 1
        AnalyticLightSample ls;
        if (!sampleLight(sd.posW, gScene.getLight(X.light), sg, ls))
            return phat;

        dir = ls.dir;
        distance = ls.distance;
        Li = ls.Li;
#else
        return phat;
#endif
    }
    else if (X.type == 2)
    {
#if USE_ENV_LIGHT == 1
        dir = X.dir;
        distance = kRayTMax;
        Li = gEnvMapSampler.eval(dir);
#else
        return phat;
#endif
    }
    else if (X.type == 3)
    {
#if USE_EMISSIVE_LIGHTS == 1
        if (X.emissive.triangleIndex >= gScene.lightCollection.getTriangleCount())
            return phat;
        var triangle = gScene.lightCollection.getTriangle(X.emissive.triangleIndex);
        var barycentric = X.emissive.getFullBarycentric();
        var uv = triangle.getTexCoord(barycentric);
        var initialPos = triangle.getPosition(barycentric);
        float distSqr = dot(initialPos - sd.posW, initialPos - sd.posW);
        float cosine = max(0, dot(-normalizeSafe(initialPos - sd.posW), triangle.normal));
        Li = gScene.materials.evalEmissive(triangle.materialID, uv) * cosine / distSqr;

        // Use points with offsets to avoid self intersection
        var safePos = computeRayOrigin(initialPos, triangle.normal);
        var origin = computeRayOrigin(sd.posW, dot(sd.faceN, safePos - sd.posW) >= 0.f ? sd.faceN : -sd.faceN);
        dir = normalizeSafe(safePos - origin);
        distance = length(safePos - origin);
#else
        return phat;
#endif
    }
    else
    {
        return phat;
    }

    // Reject sample if not in the hemisphere of a BSDF lobe.
    const uint lobeTypes = mi.getLobeTypes(sd);
    const bool hasReflection = lobeTypes & uint(LobeType::Reflection);
    const bool hasTransmission = lobeTypes & uint(LobeType::Transmission);
    float NdotL = dot(sd.getOrientedFaceNormal(), dir);
    if ((NdotL <= kMinCosTheta && !hasTransmission) || (NdotL >= -kMinCosTheta && !hasReflection))
        return phat;

    if (checkVisibility)
    {
        // Get origin with offset applied in direction of the geometry normal to avoid self-intersection.
        const float3 origin = computeRayOrigin(sd.posW, dot(sd.faceN, dir) >= 0.f ? sd.faceN : -sd.faceN);

        // Test visibility by tracing a shadow ray.
        const Ray ray = Ray(origin, dir, 0.f, distance);
        if (!traceSceneVisibilityRay<1>(ray, RAY_FLAG_NONE, 0xff))
            return phat;
    }

    // Evaluate contribution.
    var bsdf = mi.eval(sd, dir, sg);
    phat.full = (any(isnan(bsdf)) ? 0 : bsdf) * Li;
    if (gReuseDemodulated)
    {
        var props = mi.getProperties(sd);
        var albedo = props.diffuseReflectionAlbedo + props.diffuseTransmissionAlbedo + props.specularReflectionAlbedo +
                     props.specularTransmissionAlbedo;
        phat.full = phat.full / max(albedo, 1e-3);
    }
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
#if USE_ANALYTIC_LIGHTS == 1
        for (int i = 0; i < gAnalyticalSamples; ++i)
        {
            sampleAnalyticalLight(sg, X, invPdf);
            var phat = evalF(point.sd, point.mi, sg, X, gCandidatesVisibility);
            builder.update(Reservoir(X, invPdf, 1), 1.0f / gAnalyticalSamples, phat, sg);
        }
#endif
#if USE_ENV_LIGHT == 1
        for (int i = 0; i < gEnvironmentSamples; ++i)
        {
            sampleEnvMap(sg, X, invPdf);
            var phat = evalF(point.sd, point.mi, sg, X, gCandidatesVisibility);
            builder.update(Reservoir(X, invPdf, 1), 1.0f / gEnvironmentSamples, phat, sg);
        }
#endif
#if USE_EMISSIVE_LIGHTS == 1
        for (int i = 0; i < gEmissiveSamples; ++i)
        {
            sampleEmissiveLight(point.sd, sg, X, invPdf);
            var phat = evalF(point.sd, point.mi, sg, X, gCandidatesVisibility);
            builder.update(Reservoir(X, invPdf, 1), 1.0f / gEmissiveSamples, phat, sg);
        }
#endif
        if (!gCandidatesVisibility)
        {
            var candidate = builder.build();
            var candidatePhat = evalF(point.sd, point.mi, sg, candidate.Y, true);
            if (candidatePhat.phat == 0)
            {
                builder = Reservoir::Builder();
                builder.c = candidate.c;
                candidate.nullify();
            }
        }
    }

    // Normalize confidence scale so that 1 is one frame of candidates
    builder.c = 1;
    return builder;
}

struct Output
{
    float3 color;
    float3 delta;
    float3 albedo;
}
Output restirDirect(const uint2 pixel)
{
    Output output = Output(float3(0.f), float3(0.f));

    ReSTIRPoint point;
    if (point.initialize(pixel))
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
        GeometryGrads grads = GeometryGrads(pixel, point.sd.posW, z, normal, point.sd.faceN);

        // ReSTIR
        if (gIteration == 0) // Candidate + Temporal
        {
            Reservoir::Builder candidateBuilder = getCandidate(point, sg);
            var candidate = candidateBuilder.build();

            uint2 oldPixel = reproject(pixel);
            Temporal old = Temporal();
            if (gTemporalReuse && grads.allowTemporalReuse(pixel, oldPixel, z, normal))
            {
                old = gPrevReSTIR[toIndex(oldPixel)];
            }

            Phat phatOld = evalF(point.sd, point.mi, sg, old.r.Y, true);
            var builder = Reservoir::Builder();
            float mc = (float)candidate.c / (candidate.c + old.r.c);
            float wc = builder.update(candidate, mc, candidateBuilder.phat, sg);
            float wo = builder.update(old.r, 1 - mc, phatOld, sg);
            point.r = builder.build();
            LightColor l;
            {
                float3 oldColor = gTemporalColorEstimate == TemporalColorEstimate::None ? phatOld.full * old.r.W : old.c.color;
                if (gTemporalColorEstimate == TemporalColorEstimate::Gradient)
                {
                    float3 currentRaw = phatOld.full * old.r.W;
                    float3 current = LightColor(currentRaw).normalize(gNormalizeColorEstimate).color;
                    float3 prev = old.old.normalize(gNormalizeColorEstimate).color;
                    float3 largest = max(prev, current);
                    float3 relativeError = abs(prev - current) / select(largest == 0, 1, largest);
                    oldColor = lerp(oldColor, currentRaw, relativeError);
                }

                if (gNormalizeColorEstimate)
                {
                    l = LightColor::combine(LightColor(candidateBuilder.phat.full).normalize(), wc, LightColor(oldColor).normalize(), wo)
                            .normalize();
                }
                else
                {
                    l = LightColor(lerp(candidateBuilder.phat.full * candidate.W, oldColor, 1 - mc));
                }
            }
            gReSTIR[toIndex(pixel)] = Temporal(point.r, l, LightColor(phatOld.full * point.r.W));
        }
        else // Spatial + Channel
        {
            // Spatial
            ReSTIRPoint points[kSpatialReuse + 1];
            points[0] = point;
            points[0].r = gPrevReSTIR[toIndex(pixel)].r;
            points[0].c = gPrevReSTIR[toIndex(pixel)].c;
            var old = gPrevReSTIR[toIndex(pixel)].old;
            int pointsCount = 1;
            for (int i = 0; i < gMaxSpatialSearch && pointsCount <= kSpatialReuse; ++i)
            {
                float2 offset = gSpatialRadius * (sampleNext2D(sg) * 2 - 1);
                uint2 neighbor = pixel + int2(offset + sign(offset));
                if (isInBounds(neighbor))
                {
                    if (points[pointsCount].initialize(neighbor))
                    {
                        if (grads.allowSpatialReuse(pixel, neighbor, z, normal))
                        {
                            points[pointsCount].r = gPrevReSTIR[toIndex(neighbor)].r;
                            points[pointsCount].c = gPrevReSTIR[toIndex(neighbor)].c;
                            ++pointsCount;
                        }
                    }
                }
            }
            if (pointsCount > 1)
            {
                var builder = Reservoir::Builder();
                float ws[2];
                float3 fulls[2];
                float Ws[2];
                float ms[2];
                for (int i = 0; i < pointsCount; ++i)
                {
                    Phat phat = evalF(points[0].sd, points[0].mi, sg, points[i].r.Y, true);
                    float mNumerator = 0.f;
                    float mDenominator = 0.f;
                    for (int j = 0; j < pointsCount; ++j)
                    {
                        float otherPhat =
                            j == 0 ? phat.phat
                                   : (i == j ? points[i].r.phat : evalF(points[j].sd, points[j].mi, sg, points[i].r.Y, true).phat);
                        var value = points[j].r.c * otherPhat;
                        mDenominator += value;
                        if (i == j)
                            mNumerator += value;
                    }
                    float m = mDenominator == 0 ? 0 : mNumerator / mDenominator;

                    float w = builder.update(points[i].r, m, phat, sg);
                    ws[i] = w;
                    fulls[i] = phat.full;
                    Ws[i] = points[i].r.W;
                    ms[i] = m;
                }
                // Don't increase confidence during spatial reuse
                builder.c = points[0].r.c;
                points[0].r = builder.build();
                if (gNormalizeColorEstimate)
                {
                    points[0].c = LightColor::combine(points[0].c, ws[0], LightColor(fulls[1]).normalize(), ws[1]).normalize();
                }
                else
                {
                    points[0].c = LightColor::combine(points[0].c, ms[0], LightColor(fulls[1] * Ws[1]), ms[1]);
                }
                old = LightColor(builder.phat.full * points[0].r.W);
            }
            gReSTIR[toIndex(pixel)] = Temporal(points[0].r, points[0].c, old);

            {
                float3 delta = float3(0);
                for (int i = 0; i < gDeltaSamples; ++i)
                {
                    delta += evalDelta(point.sd, point.mi, sg) / gDeltaSamples;
                }
                var props = point.mi.getProperties(point.sd);
                output.color = props.emission;
                output.delta = delta;
                output.albedo = props.diffuseReflectionAlbedo + props.diffuseTransmissionAlbedo + props.specularReflectionAlbedo +
                                props.specularTransmissionAlbedo;
            }
            float3 modulate = gReuseDemodulated ? max(output.albedo, 1e-3) : float3(1);
            switch (gOutputMode)
            {
            case OutputMode::Default:
                output.color += evalF(points[0].sd, points[0].mi, sg, points[0].r.Y, true).full * points[0].r.W * modulate;
                break;
            case OutputMode::ColorDistribution:
                output.color = points[0].c.color;
                break;
            case OutputMode::Luminance:
                output.color = evalF(points[0].sd, points[0].mi, sg, points[0].r.Y, true).phat * points[0].r.W;
                break;
            case OutputMode::Combined:
                if (gNormalizeColorEstimate)
                {
                    output.color +=
                        points[0].c.apply(evalF(points[0].sd, points[0].mi, sg, points[0].r.Y, true).phat * points[0].r.W) * modulate;
                }
                else
                {
                    output.color += points[0].c.color * modulate;
                }
                break;
            default:
                output.color = float3(0.f);
                break;
            }
        }
    }
    else
    {
        gReSTIR[toIndex(pixel)] = Temporal();
        // Background pixel.
        const float3 primaryRayDir = getPrimaryRayDir(pixel, gScene.camera);
        output.color = kUseEnvBackground ? gScene.envMap.eval(primaryRayDir) : kDefaultBackgroundColor;
        output.albedo = float3(1.f, 1.f, 1.f);
    }

    return output;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (isInBounds(pixel))
    {
        Output output = restirDirect(pixel);
        gOutputColor[pixel] = float4(output.color, 1.f);
        gOutputDelta[pixel] = float4(output.delta, 1.f);
        gOutputAlbedo[pixel] = float4(output.albedo, 1.f);
    }
}
