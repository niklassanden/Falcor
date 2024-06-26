from pathlib import WindowsPath, PosixPath
from falcor import *

def render_graph_ColorReSTIR():
    g = RenderGraph('ColorReSTIR')
    g.create_pass('ToneMapper', 'ToneMapper', {'outputSize': 'Default', 'useSceneMetadata': True, 'exposureCompensation': 0.0, 'autoExposure': False, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': 'Aces', 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137, 'fNumber': 1.0, 'shutter': 1.0, 'exposureMode': 'AperturePriority'})
    g.create_pass('ColorReSTIR', 'ColorReSTIR', {'gOutputMode': 'Combined', 'gDemodulateOutput': False, 'gTemporalColorEstimate': 'Full', 'gNormalizeColorEstimate': False, 'gReuseDemodulated': False, 'gAnalyticalSamples': 1, 'gEnvironmentSamples': 1, 'gEmissiveSamples': 1, 'gDeltaSamples': 1, 'gCandidatesVisibility': False, 'gMaxConfidence': 20, 'gTemporalReuse': True, 'SPATIAL_REUSE': 1, 'gMaxSpatialSearch': 10, 'gSpatialRadius': 20})
    g.create_pass('AccumulatePass', 'AccumulatePass', {'enabled': False, 'outputSize': 'Default', 'autoReset': True, 'precisionMode': 'Single', 'maxFrameCount': 0, 'overflowMode': 'Stop'})
    g.create_pass('GBufferRT', 'GBufferRT', {'outputSize': 'Default', 'samplePattern': 'Center', 'sampleCount': 16, 'useAlphaTest': True, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': 'Back', 'texLOD': 'Mip0', 'useTraceRayInline': False, 'useDOF': True})
    g.create_pass('FileIO', 'FileIO', {})
    g.create_pass('ModulateIllumination', 'ModulateIllumination', {'useEmission': True, 'useDiffuseReflectance': False, 'useDiffuseRadiance': True, 'useSpecularReflectance': False, 'useSpecularRadiance': False, 'useDeltaReflectionEmission': False, 'useDeltaReflectionReflectance': False, 'useDeltaReflectionRadiance': True, 'useDeltaTransmissionEmission': False, 'useDeltaTransmissionReflectance': False, 'useDeltaTransmissionRadiance': False, 'useResidualRadiance': False, 'outputSize': 'Default'})
    g.add_edge('FileIO.src', 'AccumulatePass.input')
    g.add_edge('GBufferRT.viewW', 'ColorReSTIR.viewW')
    g.add_edge('GBufferRT.guideNormalW', 'ColorReSTIR.guideNormalW')
    g.add_edge('AccumulatePass.output', 'ToneMapper.src')
    g.add_edge('GBufferRT.linearZ', 'ColorReSTIR.linearZ')
    g.add_edge('GBufferRT.mvec', 'ColorReSTIR.mvec')
    g.add_edge('GBufferRT.vbuffer', 'ColorReSTIR.vbuffer')
    g.add_edge('GBufferRT.posW', 'ColorReSTIR.posW')
    g.add_edge('GBufferRT.emissive', 'ModulateIllumination.emission')
    g.add_edge('ColorReSTIR.colorHit', 'ModulateIllumination.diffuseRadiance')
    g.add_edge('ColorReSTIR.albedo', 'ModulateIllumination.diffuseReflectance')
    g.add_edge('ColorReSTIR.albedo', 'ModulateIllumination.deltaReflectionReflectance')
    g.add_edge('ColorReSTIR.delta', 'ModulateIllumination.deltaReflectionRadiance')
    g.add_edge('ModulateIllumination.output', 'FileIO.src')
    g.mark_output('ToneMapper.dst')
    return g

ColorReSTIR = render_graph_ColorReSTIR()
try: m.addGraph(ColorReSTIR)
except NameError: None
