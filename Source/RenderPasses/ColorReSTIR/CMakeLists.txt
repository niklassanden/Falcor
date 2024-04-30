add_plugin(ColorReSTIR)

target_sources(ColorReSTIR PRIVATE
    ColorReSTIR.cpp
    ColorReSTIR.h
    ColorReSTIR.rt.slang
)

target_copy_shaders(ColorReSTIR RenderPasses/ColorReSTIR)

target_source_group(ColorReSTIR "RenderPasses")
