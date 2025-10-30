#include "NeutronTracking_GPU.h"

__device__ CDParticleState_GPU* GetPSPtr() {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	return ParticleStates.ptr(ParticleStates_map[tid]);
}

__device__ int GetPSIndex() {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	return ParticleStates_map[tid];
}

__device__ int GetPSIndex(CDEventType eEventType) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	switch (eEventType) {
	case CDEventType::cross:
		return ParticlesEventState_GPU.NeuCrossSrc[tid];
		break;
	case CDEventType::colli:
		return ParticlesEventState_GPU.NeuColliSrc[tid];
		break;
	default:
		printf("GPU Error: Unknown event type!");
		return -1;
	}
}
