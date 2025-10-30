#ifndef __GPU_ConstVar_H__
#define __GPU_ConstVar_H__

#include "GPUType.h"
#include"ParaTally_GPU.h"
#include"RNG_GPU.h"
class CDSurface_GPU;
class CDSingleCell_GPU;
class CDUniverse_GPU;
class CDFissBank_GPU;
class CDIndex_GPU;
class CDCriticality_GPU;
class CDAceData_GPU;
class CDMaterial_GPU;
class CDGeometry_GPU;
class CDTempXs_GPU;
class CDErgDistr_GPU;
class CDEquiprobErgBinsLaw_GPU;
class CDInelScattLaw_GPU;
class CDContTabLaw_GPU;
class CDGenEvapLaw_GPU;
class CDMaxwellFisLaw_GPU;
class CDEvapLaw_GPU;
class CDWattLaw_GPU;
class CDTabLinearFunLaw_GPU;
class CDEquiprobErgMultiLaw_GPU;
class CDNBodyLaw_GPU;
class CDAngErgCorrKalbach87Law_GPU;
class CDAngErgCorrTabLaw_GPU;
class CDLabAngErgCorrLaw_GPU;
class CDParticleState_GPU;
class CDParticlesEventState_GPU;
class CDTally_GPU;

	////////////// �����˱���[var] ////////////////
/**
 * @brief �����ƫ����
 */
extern size_t seedOffset;
/**
 * @brief �ѱ��ߴ�
 */
extern int nFissionBankSize_host;
////////////// �����˳���[constVar] ////////////////
/**
 * @brief �����ƫ�Ƴ���
 */
const size_t seedOffsetPerCycle = 1000;
/**
 * @brief ÿ���߳̿�ĳߴ磬Ҫ��[32,1024]����Ϊ32��������Ĭ��64
 */
extern int blockSize;
/**
 * @brief �豸�˲��м�������������ӦΪWarpSize����������Ĭ�ϵ���blockSize
 */
extern int ParaTallyNum;
/**
 * @brief �豸SP����
 */
extern int NumOfSP;

////////////// �豸�˱���[var] ////////////////
/**
 * @brief ����������
 */
extern __constant__ int MaxIter;
/**
 * @brief �ѱ䷴Ӧ�����
 */
extern __constant__ int nFissMT[5];

/**
 * @brief �豸�˲��м�������������ӦΪWarpSize��������������ȡ64
 */
Dextern_constVar(ParaTallyNum_GPU, int)
/**
 * @brief �豸�����������
 */
Dextern_constVar(seed_GPU, size_t)
/**
* @brief �豸�������ƫ����
*/
Dextern_constVar(seedOffset_GPU, size_t)

Dextern_vector(rnRecord_GPU, double);
Dextern_vector(rnStartPos_GPU, int);
Dextern_vector(rnCount_GPU, int);
/**
* @brief ��������
*/
Dextern_constVar(cBoolConstVar_GPU, BoolConstVar)
/**
* @brief NeutronTransport.h�еĳ���
*/
Dextern_devVar(dStartWgt_GPU, double)
Dextern_constVar(dWgtCutOff_GPU, double)
Dextern_constVar(dEg0CutOff_GPU, double)

Dextern_constVar(ENDF_CUT_ERG_Low_GPU, double)
Dextern_constVar(INDEX_NONE_GPU, int)

Dextern_constVar(dZeroSenseDistance_GPU, double)
Dextern_constVar(dZeroMoveDistance_GPU, double)
Dextern_constVar(nVirtualMoveMaxCount_GPU, int)
Dextern_constVar(nMaxLayerNumBeforeLogging_GPU, int)
Dextern_devVar(dNeutronStartWgt_GPU, double)

////////////// �豸������[vector] ////////////////
/**
* @brief �����ѱ�����
*/
Dextern_vector(vFissParaBankCount_GPU, unsigned long long)
Dextern_vector(vFissParaBank_GPU, CDFissBank_GPU)
Dextern_vector(vFissParaSrc_GPU, CDFissBank_GPU)
Dextern_vector(vFissParaBankFlag_GPU, bool)

/**
* @brief Criticality����
*/
Dextern_devVar(cCriticality, CDCriticality_GPU)

/**
* @brief AceData����
*/
Dextern_devVar(cAceData, CDAceData_GPU)

/**
* @brief Material����
*/
Dextern_devVar(cMaterial, CDMaterial_GPU)

/**
* @brief Geometry����
*/
Dextern_devVar(cGeometry, CDGeometry_GPU)
Dextern_devVar(CellIndex_GPU, CDIndex_GPU)
Dextern_devVar(SurfaceIndex_GPU, CDIndex_GPU)
Dextern_devVar(UniverseIndex_GPU, CDIndex_GPU)
Dextern_vector(p_vSurface_GPU, CDSurface_GPU)
Dextern_vector(p_vCell_GPU, CDSingleCell_GPU)
Dextern_vector(p_vUniverse_GPU, CDUniverse_GPU)

/**
* @brief Tally����
*/
Dextern_devVar(TallyKeffCol_GPU, ParaTally<double>)
Dextern_devVar(TallyKeffAbs_GPU, ParaTally<double>)
Dextern_devVar(TallyKeffTrk_GPU, ParaTally<double>)
Dextern_devVar(cTally_GPU, CDTally_GPU)

/**
* @brief REP Control����
*/
Dextern_vector(CountReactionType, int)
Dextern_vector(ParticleStates_ReactionTypeKey, int)
Dextern_vector(ParticleStates, CDParticleState_GPU)
Dextern_vector(ParticleStates_map, int)
Dextern_vector(ParticleStates_event, int)
Dextern_vector(ParticleStates_alive, bool)
Dextern_devVar(NeuNumAlive_GPU, int)
Dextern_devVar(NeuNumAlive2_GPU, int)
Dextern_vector(ParticleStates_OnSurf, bool)
Dextern_devVar(ParticlesEventState_GPU, CDParticlesEventState_GPU)

/**
 * @brief ȫ�������������
 */
Dextern_vector(RNGs, GPURNG)

/**
* @brief dev_vectorPlus����
*/
Dextern_devVar(vectorPlusPreMallocSize_GPU, int);
Dextern_devVar(vectorPlusNum_GPU, int);
Dextern_vector(p_vCells_temp_GPU, dev_vectorPlus<int>);
Dextern_vector(p_vLocUnivs_temp_GPU, dev_vectorPlus<int>);
Dextern_vector(p_vLocCells_temp_GPU, dev_vectorPlus<int>);
Dextern_vector(p_vLocCellsU_temp_GPU, dev_vectorPlus<int>);
Dextern_vector(CrossSurfIds_temp_GPU, dev_vectorPlus<int>);
Dextern_vector(CrossSurfLocIds_temp_GPU, dev_vectorPlus<int>);
Dextern_vector(ONucCs_temp_GPU, dev_vectorPlus<CDTempXs_GPU>);
Dextern_vector(IsNucLocCellTmpChanged_temp_GPU, dev_vectorPlus<bool>);
Dextern_vector(vectorPlusStack_int_GPU, dev_vectorPlus<int>);
Dextern_vector(vectorPlusStackTemp_int_GPU, dev_vectorPlus<int>);

/**
* @brief �麯��ָ����ر��������ڻ�ȡ�豸�˵��麯��ָ��
*/
Dextern_devVar(virtualClassBase_ErgDistr, CDErgDistr_GPU);
Dextern_devVar(virtualClassBase_EquiprobErgBinsLaw, CDEquiprobErgBinsLaw_GPU);
Dextern_devVar(virtualClassBase_InelScattLaw, CDInelScattLaw_GPU);
Dextern_devVar(virtualClassBase_ContTabLaw, CDContTabLaw_GPU);
Dextern_devVar(virtualClassBase_GenEvapLaw, CDGenEvapLaw_GPU);
Dextern_devVar(virtualClassBase_MaxwellFisLaw, CDMaxwellFisLaw_GPU);
Dextern_devVar(virtualClassBase_EvapLaw, CDEvapLaw_GPU);
Dextern_devVar(virtualClassBase_WattLaw, CDWattLaw_GPU);
Dextern_devVar(virtualClassBase_TabLinearFunLaw, CDTabLinearFunLaw_GPU);
Dextern_devVar(virtualClassBase_EquiprobErgMultiLaw, CDEquiprobErgMultiLaw_GPU);
Dextern_devVar(virtualClassBase_NBodyLaw, CDNBodyLaw_GPU);
Dextern_devVar(virtualClassBase_AngErgCorrKalbach87Law, CDAngErgCorrKalbach87Law_GPU);
Dextern_devVar(virtualClassBase_AngErgCorrTabLaw, CDAngErgCorrTabLaw_GPU);
Dextern_devVar(virtualClassBase_LabAngErgCorrLaw, CDLabAngErgCorrLaw_GPU);
extern __device__ void** virtualClassPtr;





#endif // !__GPU_ConstVar_H__

