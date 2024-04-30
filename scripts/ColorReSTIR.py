from pathlib import WindowsPath, PosixPath
from falcor import *

def render_graph_ColorReSTIR():
    g = RenderGraph('ColorReSTIR')
    g.create_pass('ToneMapper', 'ToneMapper', {'outputSize': 'Default', 'useSceneMetadata': True, 'exposureCompensation': 0.0, 'autoExposure': False, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': 'Aces', 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137, 'fNumber': 1.0, 'shutter': 1.0, 'exposureMode': 'AperturePriority'})
    g.create_pass('VBufferRT', 'VBufferRT', {'outputSize': 'Default', 'samplePattern': 'Stratified', 'sampleCount': 16, 'useAlphaTest': True, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': 'Back', 'useTraceRayInline': False, 'useDOF': True})
    g.create_pass('ColorReSTIR', 'ColorReSTIR', {'maxBounces': 3, 'computeDirect': True, 'useImportanceSampling': True})
    g.create_pass('AccumulatePass', 'AccumulatePass', {'enabled': False, 'outputSize': 'Default', 'autoReset': True, 'precisionMode': 'Single', 'maxFrameCount': 0, 'overflowMode': 'Stop'})
    g.add_edge('VBufferRT.vbuffer', 'ColorReSTIR.vbuffer')
    g.add_edge('VBufferRT.mvec', 'ColorReSTIR.mvec')
    g.add_edge('VBufferRT.viewW', 'ColorReSTIR.viewW')
    g.add_edge('ColorReSTIR.color', 'AccumulatePass.input')
    g.add_edge('AccumulatePass.output', 'ToneMapper.src')
    g.mark_output('ToneMapper.dst')
    return g

ColorReSTIR = render_graph_ColorReSTIR()
try: m.addGraph(ColorReSTIR)
except NameError: None
