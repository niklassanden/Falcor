from pathlib import WindowsPath, PosixPath
from falcor import *

def render_graph_ColorReSTIRNRD():
    g = RenderGraph('ColorReSTIRNRD')
    g.create_pass('ColorReSTIR', 'ColorReSTIR', {'gOutputMode': 'Combined', 'gDemodulateOutput': False, 'gTemporalColorEstimate': 'Gradient', 'gNormalizeColorEstimate': False, 'gReuseDemodulated': False, 'gAnalyticalSamples': 1, 'gEnvironmentSamples': 1, 'gEmissiveSamples': 1, 'gDeltaSamples': 1, 'gCandidatesVisibility': False, 'gMaxConfidence': 20, 'gTemporalReuse': True, 'SPATIAL_REUSE': 1, 'gMaxSpatialSearch': 10, 'gSpatialRadius': 20})
    g.create_pass('GBufferRT', 'GBufferRT', {'outputSize': 'Default', 'samplePattern': 'Center', 'sampleCount': 16, 'useAlphaTest': True, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': 'Back', 'texLOD': 'Mip0', 'useTraceRayInline': False, 'useDOF': True})
    g.create_pass('NRD', 'NRD', {'enabled': True, 'method': 'RelaxDiffuseSpecular', 'outputSize': 'Default', 'worldSpaceMotion': True, 'disocclusionThreshold': 2.0, 'maxIntensity': 1000.0, 'diffusePrepassBlurRadius': 16.0, 'specularPrepassBlurRadius': 16.0, 'diffuseMaxAccumulatedFrameNum': 31, 'specularMaxAccumulatedFrameNum': 31, 'diffuseMaxFastAccumulatedFrameNum': 2, 'specularMaxFastAccumulatedFrameNum': 2, 'diffusePhiLuminance': 2.0, 'specularPhiLuminance': 1.0, 'diffuseLobeAngleFraction': 0.800000011920929, 'specularLobeAngleFraction': 0.8999999761581421, 'roughnessFraction': 0.5, 'diffuseHistoryRejectionNormalThreshold': 0.0, 'specularVarianceBoost': 1.0, 'specularLobeAngleSlack': 10.0, 'disocclusionFixEdgeStoppingNormalPower': 8.0, 'disocclusionFixMaxRadius': 32.0, 'disocclusionFixNumFramesToFix': 4, 'historyClampingColorBoxSigmaScale': 2.0, 'spatialVarianceEstimationHistoryThreshold': 4, 'atrousIterationNum': 6, 'minLuminanceWeight': 0.0, 'depthThreshold': 0.019999999552965164, 'luminanceEdgeStoppingRelaxation': 0.5, 'normalEdgeStoppingRelaxation': 0.30000001192092896, 'roughnessEdgeStoppingRelaxation': 0.30000001192092896, 'enableAntiFirefly': False, 'enableReprojectionTestSkippingWithoutMotion': False, 'enableSpecularVirtualHistoryClamping': False, 'enableRoughnessEdgeStopping': True, 'enableMaterialTestForDiffuse': False, 'enableMaterialTestForSpecular': False})
    g.create_pass('ToneMapper', 'ToneMapper', {'outputSize': 'Default', 'useSceneMetadata': True, 'exposureCompensation': 0.0, 'autoExposure': False, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': 'Aces', 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137, 'fNumber': 1.0, 'shutter': 1.0, 'exposureMode': 'AperturePriority'})
    g.create_pass('AccumulatePass', 'AccumulatePass', {'enabled': False, 'outputSize': 'Default', 'autoReset': True, 'precisionMode': 'Single', 'maxFrameCount': 0, 'overflowMode': 'Stop'})
    g.create_pass('ModulateIllumination', 'ModulateIllumination', {'useEmission': True, 'useDiffuseReflectance': False, 'useDiffuseRadiance': True, 'useSpecularReflectance': False, 'useSpecularRadiance': False, 'useDeltaReflectionEmission': False, 'useDeltaReflectionReflectance': False, 'useDeltaReflectionRadiance': True, 'useDeltaTransmissionEmission': False, 'useDeltaTransmissionReflectance': False, 'useDeltaTransmissionRadiance': False, 'useResidualRadiance': False, 'outputSize': 'Default'})
    g.add_edge('GBufferRT.linearZ', 'ColorReSTIR.linearZ')
    g.add_edge('AccumulatePass.output', 'ToneMapper.src')
    g.add_edge('ColorReSTIR.colorHit', 'NRD.diffuseRadianceHitDist')
    g.add_edge('GBufferRT.normWRoughnessMaterialID', 'NRD.normWRoughnessMaterialID')
    g.add_edge('GBufferRT.mvecW', 'NRD.mvec')
    g.add_edge('GBufferRT.linearZ', 'NRD.viewZ')
    g.add_edge('GBufferRT.guideNormalW', 'ColorReSTIR.guideNormalW')
    g.add_edge('GBufferRT.posW', 'ColorReSTIR.posW')
    g.add_edge('GBufferRT.viewW', 'ColorReSTIR.viewW')
    g.add_edge('GBufferRT.mvec', 'ColorReSTIR.mvec')
    g.add_edge('GBufferRT.vbuffer', 'ColorReSTIR.vbuffer')
    g.add_edge('ColorReSTIR.colorHit', 'NRD.specularRadianceHitDist')
    g.add_edge('ColorReSTIR.albedo', 'ModulateIllumination.diffuseReflectance')
    g.add_edge('NRD.filteredDiffuseRadianceHitDist', 'ModulateIllumination.diffuseRadiance')
    g.add_edge('ModulateIllumination.output', 'AccumulatePass.input')
    g.add_edge('GBufferRT.emissive', 'ModulateIllumination.emission')
    g.add_edge('ColorReSTIR.delta', 'ModulateIllumination.deltaReflectionRadiance')
    g.add_edge('ColorReSTIR.albedo', 'ModulateIllumination.deltaReflectionReflectance')
    g.mark_output('ToneMapper.dst')
    return g

ColorReSTIRNRD = render_graph_ColorReSTIRNRD()
try: m.addGraph(ColorReSTIRNRD)
except NameError: None
