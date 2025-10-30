#ifndef __GPU_BASEFUNC_H__
#define  __GPU_BASEFUNC_H__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ const char* strrchr_GPUd(const char* str, int c);
__device__ char* strrchr_GPUd(char* str, int c);

__device__ char* uint64_to_string(uint64_t value, char* str, int base);
__device__ char* to_string(size_t value, char* str);
__device__ char* strMerge(const char* str1, const char* str2, char* output);
__device__ char* indexInfoToString(size_t a1, size_t a2, char* output);
#endif //__GPU_BASEFUNC_H__