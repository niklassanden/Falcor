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
#include "ColorReSTIR.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ColorReSTIR>();
}

namespace
{
const char kShaderFile[] = "RenderPasses/ColorReSTIR/ColorReSTIR.rt.slang";

// Ray tracing settings that affect the traversal stack size.
// These should be set as small as possible.
const uint32_t kMaxPayloadSizeBytes = 72u;
const uint32_t kMaxRecursionDepth = 2u;

const char kInputViewDir[] = "viewW";
const char kNormals[] = "guideNormalW";
const char kLinearZ[] = "linearZ";
const char kPosW[] = "posW";
const ChannelList kInputChannels = {
    // clang-format off
    { "vbuffer",        "gVBuffer",         "Visibility buffer in packed format" },
    { "mvec",           "gMVec",            "Motion vectors" },
    { kNormals,         "gNormals",         "Guide normals in world space" },
    { kLinearZ,         "gLinearZ",         "Linear depth and its derivative" },
    { kPosW,            "gPosW",            "World position" },
    { kInputViewDir,    "gViewW",           "World-space view direction (xyz float format)", true /* optional */ },
    // clang-format on
};

const char kPrevNormals[] = "prevGuideNormalW";
const char kPrevLinearZ[] = "prevLinearZ";
const char kPrevPosW[] = "prevPosW";
const ChannelList kInternalChannels = {
    // clang-format off
    { kPrevNormals,     "gPrevNormals",     "Guide normals in world space from the last frame", false, ResourceFormat::RGBA32Float },
    { kPrevLinearZ,     "gPrevLinearZ",     "LinearZ from the last frame", false, ResourceFormat::RG32Float },
    { kPrevPosW,        "gPrevPosW",        "World position from the last frame", false, ResourceFormat::RGBA32Float },
    // clang-format on
};

const ChannelList kOutputChannels = {
    // clang-format off
    { "color",          "gOutputColor",     "Output color (sum of direct and indirect)", false, ResourceFormat::RGBA32Float },
    { "albedo",         "gOutputAlbedo",    "Sum of diffuse and specular reflectance", false, ResourceFormat::RGBA32Float },
    // clang-format on
};

const char kReSTIR[] = "gReSTIR";
const char kPrevReSTIR[] = "gPrevReSTIR";

const char kOutputMode[] = "gOutputMode";
const char kAnalyticalSamples[] = "gAnalyticalSamples";
const char kEnvironmentSamples[] = "gEnvironmentSamples";
const char kEmissiveSamples[] = "gEmissiveSamples";
const char kBsdfSamples[] = "gBsdfSamples";
const char kCandidatesVisibility[] = "gCandidatesVisibility";
const char kMaxConfidence[] = "gMaxConfidence";
const char kTemporalReuse[] = "gTemporalReuse";
const char kSpatialReuse[] = "SPATIAL_REUSE";
const char kMaxSpatialSearch[] = "gMaxSpatialSearch";
const char kSpatialRadius[] = "gSpatialRadius";

// MinimalPathTracer
const char kMaxBounces[] = "maxBounces";
const char kComputeDirect[] = "computeDirect";
const char kUseImportanceSampling[] = "useImportanceSampling";

struct EmissiveSample
{
    float2 uv{0, 0};
    uint triangleIndex{0};
};
struct ReSTIRSample
{
    int type{0};
    uint light{0};             // analytical
    float3 dir{0, 0, 0};       // environment
    EmissiveSample emissive{}; // emissive
};
struct LightColor
{
    float3 color{0, 0, 0};
};
struct Reservoir
{
    ReSTIRSample Y{};
    float W{0.0f};
    int c{0};
    float phat{0};
};
struct Temporal
{
    Reservoir r;
    LightColor c;
};
} // namespace

ColorReSTIR::ColorReSTIR(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    parseProperties(props);

    mSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    FALCOR_ASSERT(mSampleGenerator);
}

void ColorReSTIR::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kOutputMode)
            mConfig.outputMode = value;
        else if (key == kAnalyticalSamples)
            mConfig.analyticalSamples = value;
        else if (key == kEnvironmentSamples)
            mConfig.environmentSamples = value;
        else if (key == kEmissiveSamples)
            mConfig.emissiveSamples = value;
        else if (key == kBsdfSamples)
            mConfig.bsdfSamples = value;
        else if (key == kCandidatesVisibility)
            mConfig.candidatesVisibility = value;
        else if (key == kMaxConfidence)
            mConfig.maxConfidence = value;
        else if (key == kTemporalReuse)
            mConfig.temporalReuse = value;
        else if (key == kSpatialReuse)
            mConfig.spatialReuse = value;
        else if (key == kMaxSpatialSearch)
            mConfig.maxSpatialSearch = value;
        else if (key == kSpatialRadius)
            mConfig.spatialRadius = value;

        // MinimalPathTracer
        else if (key == kMaxBounces)
            mMaxBounces = value;
        else if (key == kComputeDirect)
            mComputeDirect = value;
        else if (key == kUseImportanceSampling)
            mUseImportanceSampling = value;

        else
            logWarning("Unknown property '{}' in ColorReSTIR properties.", key);
    }
    updateDefines();
}

Properties ColorReSTIR::getProperties() const
{
    Properties props;
    props[kOutputMode] = mConfig.outputMode;
    props[kAnalyticalSamples] = mConfig.analyticalSamples;
    props[kEnvironmentSamples] = mConfig.environmentSamples;
    props[kEmissiveSamples] = mConfig.emissiveSamples;
    props[kBsdfSamples] = mConfig.bsdfSamples;
    props[kCandidatesVisibility] = mConfig.candidatesVisibility;
    props[kMaxConfidence] = mConfig.maxConfidence;
    props[kTemporalReuse] = mConfig.temporalReuse;
    props[kSpatialReuse] = mConfig.spatialReuse;
    props[kMaxSpatialSearch] = mConfig.maxSpatialSearch;
    props[kSpatialRadius] = mConfig.spatialRadius;

    // MinimalPathTracer
    props[kMaxBounces] = mMaxBounces;
    props[kComputeDirect] = mComputeDirect;
    props[kUseImportanceSampling] = mUseImportanceSampling;

    return props;
}

RenderPassReflection ColorReSTIR::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);

    for (const auto& desc : kInternalChannels)
    {
        reflector.addInternal(desc.name, desc.desc).format(desc.format).flags(RenderPassReflection::Field::Flags::Persistent);
    }

    return reflector;
}

void ColorReSTIR::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    const size_t count = compileData.defaultTexDims.x * compileData.defaultTexDims.y;
    const std::vector<Temporal> data(count);
    const auto defaultFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    for (size_t i = 0; i < std::size(mReSTIRBuffers); ++i)
    {
        mReSTIRBuffers[i] = mpDevice->createStructuredBuffer(sizeof(Temporal), count, defaultFlags, MemoryType::DeviceLocal, data.data());
    }
}

void ColorReSTIR::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (mHotReloaded)
    {
        setScene(pRenderContext, mScene);
        mHotReloaded = false;
    }

    // Update refresh flag if options that affect the output have changed.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    // If we have no scene, just clear the outputs and return.
    if (!mScene)
    {
        clearRenderPassChannels(pRenderContext, kOutputChannels, renderData);
        return;
    }

    if (is_set(mScene->getUpdates(), Scene::UpdateFlags::RecompileNeeded) ||
        is_set(mScene->getUpdates(), Scene::UpdateFlags::GeometryChanged))
    {
        FALCOR_THROW("This render pass does not support scene changes that require shader recompilation.");
    }

    // Request the light collection if emissive lights are enabled.
    if (mScene->getRenderSettings().useEmissiveLights)
    {
        mScene->getLightCollection(pRenderContext);
    }

    // Configure depth-of-field.
    const bool useDOF = mScene->getCamera()->getApertureRadius() > 0.f;
    if (useDOF && renderData[kInputViewDir] == nullptr)
    {
        logWarning("Depth-of-field requires the '{}' input. Expect incorrect shading.", kInputViewDir);
    }

    // Specialize program.
    // These defines should not modify the program vars. Do not trigger program vars re-creation.
    mTracer.program->addDefine(kSpatialReuse, std::to_string(mDefines.spatialReuse));
    // MinimalPathTracer
    mTracer.program->addDefine("MAX_BOUNCES", std::to_string(mMaxBounces));
    mTracer.program->addDefine("COMPUTE_DIRECT", mComputeDirect ? "1" : "0");
    mTracer.program->addDefine("USE_IMPORTANCE_SAMPLING", mUseImportanceSampling ? "1" : "0");
    mTracer.program->addDefine("USE_ANALYTIC_LIGHTS", mScene->useAnalyticLights() ? "1" : "0");
    mTracer.program->addDefine("USE_EMISSIVE_LIGHTS", mScene->useEmissiveLights() ? "1" : "0");
    mTracer.program->addDefine("USE_ENV_LIGHT", mScene->useEnvLight() ? "1" : "0");
    mTracer.program->addDefine("USE_ENV_BACKGROUND", mScene->useEnvBackground() ? "1" : "0");

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    mTracer.program->addDefines(getValidResourceDefines(kInputChannels, renderData));
    mTracer.program->addDefines(getValidResourceDefines(kOutputChannels, renderData));

    if (mScene->useEnvLight())
    {
        if (!mEnvMapSampler || mEnvMapSampler->getEnvMap() != mScene->getEnvMap())
        {
            mEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, mScene->getEnvMap());
        }
    }
    if (mScene->useEmissiveLights())
    {
        if (!mEmissiveSampler)
        {
            mEmissiveSampler = std::make_unique<LightBVHSampler>(pRenderContext, mScene);
        }
        mEmissiveSampler->update(pRenderContext);
        mTracer.program->addDefines(mEmissiveSampler->getDefines());
    }

    // Prepare program vars. This may trigger shader compilation.
    // The program should have all necessary defines set at this point.
    if (!mTracer.vars)
        prepareVars();
    FALCOR_ASSERT(mTracer.vars);

    // Set constants.
    auto var = mTracer.vars->getRootVar();
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gPRNGDimension"] = dict.keyExists(kRenderPassPRNGDimension) ? dict[kRenderPassPRNGDimension] : 0u;
    var["CB"][kOutputMode] = static_cast<uint32_t>(mConfig.outputMode);
    var["CB"][kAnalyticalSamples] = mConfig.analyticalSamples;
    var["CB"][kEnvironmentSamples] = mConfig.environmentSamples;
    var["CB"][kEmissiveSamples] = mConfig.emissiveSamples;
    var["CB"][kBsdfSamples] = mConfig.bsdfSamples;
    var["CB"][kCandidatesVisibility] = mConfig.candidatesVisibility;
    var["CB"][kMaxConfidence] = mConfig.maxConfidence;
    var["CB"][kTemporalReuse] = mConfig.temporalReuse;
    var["CB"][kMaxSpatialSearch] = mConfig.maxSpatialSearch;
    var["CB"][kSpatialRadius] = mConfig.spatialRadius;

    if (mScene->useEnvLight() && mEnvMapSampler)
    {
        mEnvMapSampler->bindShaderData(var["gEnvMapSampler"]);
    }
    if (mScene->useEmissiveLights() && mEmissiveSampler)
    {
        mEmissiveSampler->bindShaderData(var["gEmissiveSampler"]);
    }

    // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            var[desc.texname] = renderData.getTexture(desc.name);
        }
    };
    for (const auto& channel : kInputChannels)
        bind(channel);
    for (const auto& channel : kOutputChannels)
        bind(channel);
    for (const auto& channel : kInternalChannels)
        bind(channel);

    // Get dimensions of ray dispatch.
    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    // Spawn the rays.
    for (int it = 0; it < 2; ++it)
    {
        var["CB"]["gIteration"] = it;
        var[kReSTIR] = mReSTIRBuffers[0];
        var[kPrevReSTIR] = mReSTIRBuffers[1];
        std::swap(mReSTIRBuffers[0], mReSTIRBuffers[1]);
        mScene->raytrace(pRenderContext, mTracer.program.get(), mTracer.vars, uint3(targetDim, 1));
    }

    pRenderContext->blit(renderData.getTexture(kNormals)->getSRV(), renderData.getTexture(kPrevNormals)->getRTV());
    pRenderContext->blit(renderData.getTexture(kLinearZ)->getSRV(), renderData.getTexture(kPrevLinearZ)->getRTV());
    pRenderContext->blit(renderData.getTexture(kPosW)->getSRV(), renderData.getTexture(kPrevPosW)->getRTV());

    mFrameCount++;
}

void ColorReSTIR::renderUI(Gui::Widgets& widget)
{
    static constexpr float intSpeed = 0.02f;
    bool dirty = false;

    if (definesOutdated())
    {
        auto pressed = widget.button("Update defines");
        widget.tooltip(
            "Updates defines and recompiles shaders (if this version is not already cached). This button is only visible if the defines "
            "our out of date.",
            true
        );
        if (pressed)
        {
            dirty = true;
            updateDefines();
        }
    }

    dirty |= widget.dropdown("Output Mode", mConfig.outputMode);

    {
        auto group = widget.group("Candidate sample counts", false);

        dirty |= group.var("Analytical", mConfig.analyticalSamples, 0u, 1u << 16, intSpeed);
        group.tooltip("Number of analytical light samples to generate.", true);

        dirty |= group.var("Environment", mConfig.environmentSamples, 0u, 1u << 16, intSpeed);
        group.tooltip("Number of environment map samples to generate.", true);

        dirty |= group.var("Emissive", mConfig.emissiveSamples, 0u, 1u << 16, intSpeed);
        group.tooltip("Number of emissive light samples to generate.", true);

        dirty |= group.var("BSDF", mConfig.bsdfSamples, 0u, 1u << 16, intSpeed);
        group.tooltip("(WARNING: One extra ray is cast for each BSDF sample)\nNumber of BSDF samples to generate.", true);
    }

    dirty |= widget.checkbox("Candidate visibility", mConfig.candidatesVisibility);
    widget.tooltip("If enabled, each candidate sample will shoot shadow rays to compute visibility.", true);

    dirty |= widget.var("Max confidence", mConfig.maxConfidence, 1u, 1u << 16, intSpeed);
    widget.tooltip("Clamps the confidence to this value. This controls the weight in the temporal accumulation.", true);

    dirty |= widget.checkbox("Temporal reuse", mConfig.temporalReuse);
    widget.tooltip("Whether or not to do temporal reuse.", true);

    dirty |= widget.var("Spatial reuse", mConfig.spatialReuse, 0u, 1u << 16, intSpeed);
    widget.tooltip(
        "(Recompiles shaders). The number of neighbors to do spatial reuse from. Note that this is an upper bound, the actual number "
        "depends on how many are found.",
        true
    );

    dirty |= widget.var("Max spatial search", mConfig.maxSpatialSearch, 0u, 1u << 16, intSpeed);
    widget.tooltip("The number of pixels we are allowed to look at when finding neighbors for spatial reuse.", true);

    dirty |= widget.var("Spatial radius", mConfig.spatialRadius, 0u, 1u << 16, intSpeed);
    widget.tooltip("The radius for spatial reuse measured in pixels.", true);

    // Minimal Path Tracer
    {
        auto group = widget.group("Minimal Path Tracer Options", false);
        dirty |= group.var("Max bounces", mMaxBounces, 0u, 1u << 16);
        group.tooltip("Maximum path length for indirect illumination.\n0 = direct only\n1 = one indirect bounce etc.", true);

        dirty |= group.checkbox("Evaluate direct illumination", mComputeDirect);
        group.tooltip("Compute direct illumination.\nIf disabled only indirect is computed (when max bounces > 0).", true);

        dirty |= group.checkbox("Use importance sampling", mUseImportanceSampling);
        group.tooltip("Use importance sampling for materials", true);
    }

    // If rendering options that modify the output have changed, set flag to indicate that.
    // In execute() we will pass the flag to other passes for reset of temporal data etc.
    if (dirty)
    {
        mOptionsChanged = true;
    }
}

void ColorReSTIR::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    // Clear data for previous scene.
    // After changing scene, the raytracing program should to be recreated.
    mTracer.program = nullptr;
    mTracer.bindingTable = nullptr;
    mTracer.vars = nullptr;
    mEnvMapSampler = nullptr;
    mEmissiveSampler = nullptr;
    mFrameCount = 0;

    // Set new scene.
    mScene = pScene;

    if (mScene)
    {
        if (pScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("ColorReSTIR: This render pass does not support custom primitives.");
        }

        // Create ray tracing program.
        ProgramDesc desc;
        desc.addShaderModules(mScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(mScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

        mTracer.bindingTable = RtBindingTable::create(2, 2, mScene->getGeometryCount());
        auto& sbt = mTracer.bindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("scatterMiss"));
        sbt->setMiss(1, desc.addMiss("shadowMiss"));

        if (mScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            sbt->setHitGroup(
                0,
                mScene->getGeometryIDs(Scene::GeometryType::TriangleMesh),
                desc.addHitGroup("scatterTriangleMeshClosestHit", "scatterTriangleMeshAnyHit")
            );
            sbt->setHitGroup(
                1, mScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("", "shadowTriangleMeshAnyHit")
            );
        }

        if (mScene->hasGeometryType(Scene::GeometryType::DisplacedTriangleMesh))
        {
            sbt->setHitGroup(
                0,
                mScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
                desc.addHitGroup("scatterDisplacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection")
            );
            sbt->setHitGroup(
                1,
                mScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
                desc.addHitGroup("", "", "displacedTriangleMeshIntersection")
            );
        }

        if (mScene->hasGeometryType(Scene::GeometryType::Curve))
        {
            sbt->setHitGroup(
                0, mScene->getGeometryIDs(Scene::GeometryType::Curve), desc.addHitGroup("scatterCurveClosestHit", "", "curveIntersection")
            );
            sbt->setHitGroup(1, mScene->getGeometryIDs(Scene::GeometryType::Curve), desc.addHitGroup("", "", "curveIntersection"));
        }

        if (mScene->hasGeometryType(Scene::GeometryType::SDFGrid))
        {
            sbt->setHitGroup(
                0,
                mScene->getGeometryIDs(Scene::GeometryType::SDFGrid),
                desc.addHitGroup("scatterSdfGridClosestHit", "", "sdfGridIntersection")
            );
            sbt->setHitGroup(1, mScene->getGeometryIDs(Scene::GeometryType::SDFGrid), desc.addHitGroup("", "", "sdfGridIntersection"));
        }

        mTracer.program = Program::create(mpDevice, desc, mScene->getSceneDefines());
    }
}

void ColorReSTIR::prepareVars()
{
    FALCOR_ASSERT(mScene);
    FALCOR_ASSERT(mTracer.program);

    // Configure program.
    mTracer.program->addDefines(mSampleGenerator->getDefines());
    mTracer.program->setTypeConformances(mScene->getTypeConformances());

    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mTracer.vars = RtProgramVars::create(mpDevice, mTracer.program, mTracer.bindingTable);

    // Bind utility classes into shared data.
    auto var = mTracer.vars->getRootVar();
    mSampleGenerator->bindShaderData(var);
}

bool ColorReSTIR::definesOutdated()
{
    return mDefines.spatialReuse != mConfig.spatialReuse;
}
void ColorReSTIR::updateDefines()
{
    mDefines.spatialReuse = mConfig.spatialReuse;
}

void ColorReSTIR::onHotReload(HotReloadFlags reloaded)
{
    mHotReloaded = true;
}
