#ifndef __RNG_GPU__H__
#define  __RNG_GPU__H__

#include <curand.h>
#include <curand_kernel.h>

#include "GPU_interface.h"
#include "GPUType.h"

// ʹ��10�������ͼ���������
typedef curandStatePhilox4_32_10_t GPURNG;

/**
 * @brief ��ʼ�������������
 */
__device__ void InitRNG(GPURNG* rng, size_t idn, size_t offset);

/**
 * @brief ��ȡ��׼�����
 * @param rng �����������ָ��
 */
__device__ double GetRN(GPURNG* rng);

/**
 * @brief ÿ�����������������ƫ����
 * @param nCyc ����
 */
void UpdateSeedOffset(int nCyc);

/**
 * @brief ��ȡ���ȷֲ������
 */
__device__ double UniformDistribution_GPUd(const double& lower, const double& upper, GPURNG* rng);


#endif