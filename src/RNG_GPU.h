#ifndef __RNG_GPU__H__
#define  __RNG_GPU__H__

#include <curand.h>
#include <curand_kernel.h>

#include "GPU_interface.h"
#include "GPUType.h"

// 使用10参数调和级数发生器
typedef curandStatePhilox4_32_10_t GPURNG;

/**
 * @brief 初始化随机数发生器
 */
__device__ void InitRNG(GPURNG* rng, size_t idn, size_t offset);

/**
 * @brief 获取标准随机数
 * @param rng 随机数发生器指针
 */
__device__ double GetRN(GPURNG* rng);

/**
 * @brief 每代更新随机数发生器偏移量
 * @param nCyc 代数
 */
void UpdateSeedOffset(int nCyc);

/**
 * @brief 获取均匀分布随机数
 */
__device__ double UniformDistribution_GPUd(const double& lower, const double& upper, GPURNG* rng);


#endif