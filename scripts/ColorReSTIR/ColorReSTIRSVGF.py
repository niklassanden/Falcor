from pathlib import WindowsPath, PosixPath
from falcor import *

def render_graph_ColorReSTIR():
    g = RenderGraph('ColorReSTIR')
    g.create_pass('GBufferRaster', 'GBufferRaster', {'outputSize': 'Default', 'samplePattern': 'Center', 'sampleCount': 16, 'useAlphaTest': True, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': 'Back'})
    g.create_pass('ToneMapper', 'ToneMapper', {'outputSize': 'Default', 'useSceneMetadata': True, 'exposureCompensation': 0.0, 'autoExposure': False, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': 'Aces', 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137, 'fNumber': 1.0, 'shutter': 1.0, 'exposureMode': 'AperturePriority'})
    g.create_pass('ColorReSTIR', 'ColorReSTIR', {'maxBounces': 3, 'computeDirect': True, 'useImportanceSampling': True, 'SPLIT_CHANNELS': False, 'gCandidateCount': 1, 'gCandidatesVisibility': False, 'gReuseCandidates': True, 'gMaxConfidence': 20, 'gTemporalReuse': True, 'SPATIAL_REUSE': 1, 'gMaxSpatialSearch': 10, 'gSpatialRadius': 20, 'gChannelReuse': True})
    g.create_pass('AccumulatePass', 'AccumulatePass', {'enabled': False, 'outputSize': 'Default', 'autoReset': True, 'precisionMode': 'Single', 'maxFrameCount': 0, 'overflowMode': 'Stop'})
    g.create_pass('GBufferRT', 'GBufferRT', {'outputSize': 'Default', 'samplePattern': 'Center', 'sampleCount': 16, 'useAlphaTest': True, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': 'Back', 'texLOD': 'Mip0', 'useTraceRayInline': False, 'useDOF': True})
    g.create_pass('GBufferRaster0', 'GBufferRaster', {'outputSize': 'Default', 'samplePattern': 'Center', 'sampleCount': 16, 'useAlphaTest': True, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': 'Back'})
    g.create_pass('SVGFPass', 'SVGFPass', {'Enabled': True, 'Iterations': 4, 'FeedbackTap': 1, 'VarianceEpsilon': 9.999999747378752e-05, 'PhiColor': 10.0, 'PhiNormal': 128.0, 'Alpha': 0.05000000074505806, 'MomentsAlpha': 0.20000000298023224})
    g.add_edge('SVGFPass.Filtered image', 'AccumulatePass.input')
    g.add_edge('GBufferRT.viewW', 'ColorReSTIR.viewW')
    g.add_edge('GBufferRT.guideNormalW', 'ColorReSTIR.guideNormalW')
    g.add_edge('AccumulatePass.output', 'ToneMapper.src')
    g.add_edge('GBufferRT.linearZ', 'ColorReSTIR.linearZ')
    g.add_edge('GBufferRT.mvec', 'ColorReSTIR.mvec')
    g.add_edge('GBufferRT.vbuffer', 'ColorReSTIR.vbuffer')
    g.add_edge('GBufferRT.posW', 'ColorReSTIR.posW')
    g.add_edge('GBufferRaster0.pnFwidth', 'SVGFPass.PositionNormalFwidth')
    g.add_edge('ColorReSTIR.color', 'SVGFPass.Color')
    g.add_edge('GBufferRaster0.emissive', 'SVGFPass.Emission')
    g.add_edge('GBufferRaster0.linearZ', 'SVGFPass.LinearZ')
    g.add_edge('ColorReSTIR.albedo', 'SVGFPass.Albedo')
    g.add_edge('GBufferRaster0.posW', 'SVGFPass.WorldPosition')
    g.add_edge('GBufferRaster0.guideNormalW', 'SVGFPass.WorldNormal')
    g.add_edge('GBufferRaster0.mvec', 'SVGFPass.MotionVec')
    g.mark_output('ToneMapper.dst')
    return g

ColorReSTIR = render_graph_ColorReSTIR()
try: m.addGraph(ColorReSTIR)
except NameError: None
