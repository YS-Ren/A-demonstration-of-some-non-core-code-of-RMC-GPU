#ifndef __NEUTRONTRACKING_GPU_H__
#define __NEUTRONTRACKING_GPU_H__

#include "GPU_interface.h"
#include "GPUConstVar.h"
#include "ParticleState_GPU.h"
#include "Material_GPU.h"
#include "AceData_GPU.h"
#include "Geometry_GPU.h"

enum class CDEventType {
    colli = 0,
    cross
};

__device__ CDParticleState_GPU* GetPSPtr();
__device__ int GetPSIndex();
__device__ int GetPSIndex(CDEventType eEventType);

__device__ void SampleFisSource_GPUd(int PSIndex);
__device__ void TrackHistory_GPUd(const int nNeu);

__device__ void RayTracking_GPUd(int PSIndex, const int nNeu);

__device__ double GetNextTrackCalFlyDistance_GPUd(int PSIndex, bool bGPT, double dTimeAbsMacroCs = 0, const bool bActive = false);

__device__ void SampleColliNuc_GPUd(int PSIndex);

__device__ void CalcColliNucCs_GPUd(int PSIndex, const bool bActive);

__device__ void CalcColliNucCs_GPUd(int PSIndex);

__device__ void TreatFission_GPUd(int PSIndex, const int nNeu, bool Iscvmt);

__device__ void BankFisSource_GPUd(int PSIndex, const double dFisSubCs[5],  const int nNeu);

__device__ void GetFissionNeuState_GPUd(int PSIndex, int nFisMT, double dFisWgt, const int nNeu);

__device__ void GetExitState_GPUd(int PSIndex);

__device__ void GetCeExitState_GPUd(int PSIndex, int nColliNuc, int nReactType, double dIncidErg, const double* dIncidDir, bool bTreatedFreeGas,
    double& dExitErg, double* dExitDir);

#ifdef AIS
__device__ void GetCeExitState_GPUd(int PSIndex, int nColliNuc, NeutronReactionType eNeuReactionType, double dIncidErg, const double* dIncidDir,
    bool bTreatedFreeGas, double& dExitErg, double* dExitDir);
#endif // AIS

__device__ void TreatImpliCapt_GPUd(int PSIndex, bool bAbsIncludeFis);

__device__ void AnalogRussianRoulette_GPUd(int PSIndex);

__device__ int SampleColliMT_Cri_GPUd(int PSIndex);

#ifdef AIS
__device__ NeutronReactionType SampleColliType_Cri_GPUd(int PSIndex);
#endif // AIS

__device__ double SampleFreeFlyDist(int PSIndex, bool IsErgChanged, double dTimeAbsMacroCs = 0, const bool bActive = false);

__device__ void CalcMacroXS(int PSIndex, bool IsErgChanged, const bool bActive = false);

__device__ void TreatFreeGasModel_GPUd(int PSIndex);

__device__ void TreatFreeGasModel_GPUd(int PSIndex, const bool bActive);

void TrackHistory_GPUh(int threadNum);

#endif