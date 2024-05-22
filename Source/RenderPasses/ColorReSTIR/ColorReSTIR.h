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
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Rendering/Lights/EnvMapSampler.h"
#include "Rendering/Lights/LightBVHSampler.h"
#include "Utils/Sampling/SampleGenerator.h"

using namespace Falcor;

/**
 * Color ReSTIR
 *
 * This pass implements ReSTIR with one reservoir per colour channel.
 *
 * Based on the MinimalPathTracer Render Pass.
 */
class ColorReSTIR : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ColorReSTIR, "ColorReSTIR", "A ReSTIR implementation with one reservoir per colour channel.");

    static ref<ColorReSTIR> create(ref<Device> pDevice, const Properties& props) { return make_ref<ColorReSTIR>(pDevice, props); }

    ColorReSTIR(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }
    virtual void onHotReload(HotReloadFlags reloaded) override;

    enum class OutputMode : uint32_t
    {
        Default = 0u,
        Luminance = 1u,
        ColorDistribution = 2u,
        Combined = 3u,
    };
    FALCOR_ENUM_INFO(
        OutputMode,
        {
            {OutputMode::Default, "Default"},
            {OutputMode::Luminance, "Luminance"},
            {OutputMode::ColorDistribution, "ColorDistribution"},
            {OutputMode::Combined, "Combined"},
        }
    );

    enum class TemporalColorEstimate : uint32_t
    {
        None = 0u,
        Full = 1u,
        Gradient = 2u,
    };
    FALCOR_ENUM_INFO(
        TemporalColorEstimate,
        {
            {TemporalColorEstimate::None, "None"},
            {TemporalColorEstimate::Full, "Full"},
            {TemporalColorEstimate::Gradient, "Gradient"},
        }
    );

private:
    void parseProperties(const Properties& props);
    bool definesOutdated();
    void updateDefines();

    // Internal state

    /// Current scene.
    ref<Scene> mScene;
    /// GPU sample generator.
    ref<SampleGenerator> mSampleGenerator;
    /// Environment map sampler.
    std::unique_ptr<EnvMapSampler> mEnvMapSampler;
    /// Emissive light sampler.
    std::unique_ptr<EmissiveLightSampler> mEmissiveSampler;
    /// Screen space reservoirs.
    /// Ping pong buffer because of the spatial reuse.
    ref<Buffer> mReSTIRBuffers[2];

    // Configuration

    struct
    {
        OutputMode outputMode = OutputMode::Combined;
        TemporalColorEstimate temporalColorEstimate = TemporalColorEstimate::Gradient;
        bool normalizeColorEstimate = false;
        bool reuseDemodulated = false;
        uint analyticalSamples = 4;
        uint environmentSamples = 4;
        uint emissiveSamples = 4;
        uint deltaSamples = 1;
        bool candidatesVisibility = false;
        uint maxConfidence = 20;
        bool temporalReuse = true;
        uint spatialReuse = 1;
        uint maxSpatialSearch = 10;
        uint spatialRadius = 20;
    } mConfig;
    struct
    {
        uint spatialReuse = 1;
    } mDefines;

    // Runtime data

    /// Frame count since scene was loaded.
    uint mFrameCount = 0;
    bool mOptionsChanged = false;
    bool mHotReloaded = false;

    /// ReSTIR Compute Pass (uses inline ray tracing)
    ref<ComputePass> mPass;
};
FALCOR_ENUM_REGISTER(ColorReSTIR::OutputMode);
FALCOR_ENUM_REGISTER(ColorReSTIR::TemporalColorEstimate);
