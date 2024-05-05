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

private:
    void parseProperties(const Properties& props);
    void prepareVars();
    bool definesOutdated();
    void updateDefines();

    // Internal state

    /// Current scene.
    ref<Scene> mScene;
    /// GPU sample generator.
    ref<SampleGenerator> mSampleGenerator;
    /// Environment map sampler.
    std::unique_ptr<EnvMapSampler> mEnvMapSampler;
    /// Screen space reservoirs.
    /// Ping pong buffer because of the spatial reuse.
    ref<Buffer> mReSTIRBuffers[2];

    // Configuration

    struct
    {
        uint candidateCount = 5;
        bool candidatesVisibility = false;
        uint maxConfidence = 20;
        bool temporalReuse = true;
        uint spatialReuse = 1;
        uint maxSpatialSearch = 10;
        uint spatialRadius = 20;
        bool channelReuse = true;
    } mConfig;
    struct
    {
        uint spatialReuse = 1;
    } mDefines;

    /// Max number of indirect bounces (0 = none).
    uint mMaxBounces = 3;
    /// Compute direct illumination (otherwise indirect only).
    bool mComputeDirect = true;
    /// Use importance sampling for materials.
    bool mUseImportanceSampling = true;

    // Runtime data

    /// Frame count since scene was loaded.
    uint mFrameCount = 0;
    bool mOptionsChanged = false;
    bool mHotReloaded = false;

    /// Ray tracing program.
    struct
    {
        ref<Program> program;
        ref<RtBindingTable> bindingTable;
        ref<RtProgramVars> vars;
    } mTracer;
};
