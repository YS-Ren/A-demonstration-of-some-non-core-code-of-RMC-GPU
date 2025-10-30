#ifndef __ParaTally_GPU_H__
#define __ParaTally_GPU_H__

#include "GPUType.h"
#include"GPUOperator.h"

template <typename T>
class ParaTally {
public:
	// 并行计数器总数
	int paraTallyNum;
	// 并行计数器计数
	int* paraTally;
	// 并行计数器数据存储库容量
	int paraBankCapacity;
	// 数据库包含元素个数;
	int paraBankCount;
	// 并行计数器数据存储数组
	T* paraBank;
	// 并行计数器数据存储数组标识符
	bool* paraBankFlag;
	//统计量总和
	T sum;

	//构造函数 
	__host__ __device__ ParaTally() = default;

	__host__ ParaTally(int tally_num, int bank_size) {
		Register(tally_num, bank_size);
	}
	__host__ void Register(int tally_num, int bank_size) {
		paraTallyNum = tally_num;
		paraBankCapacity = bank_size;
		paraBankCount = 0;

		int* paraTally_temp = new int[paraTallyNum];
		for (int i = 0; i < paraTallyNum; ++i) { paraTally_temp[i] = 0; }
		{
			CUdeviceptr ptr_temp;
			size_t size_temp = sizeof(int) * paraTallyNum;
			cudaMalloc(reinterpret_cast<void**>(&ptr_temp), size_temp);
			cudaMemcpy(reinterpret_cast<void*>(ptr_temp), paraTally_temp, size_temp, cudaMemcpyHostToDevice);
			paraTallyInt = reinterpret_cast<int*>(ptr_temp);
		}
		delete[] paraTally_temp;

		T* paraBank_temp = new T[paraBankCapacity];
		for (int i = 0; i < paraBankCapacity; ++i) { paraBank_temp[i] = 0; }
		{
			CUdeviceptr ptr_temp;
			size_t size_temp = sizeof(T) * paraBankCapacity;
			cudaMalloc(reinterpret_cast<void**>(&ptr_temp), size_temp);
			cudaMemcpy(reinterpret_cast<void*>(ptr_temp), paraBank_temp, size_temp, cudaMemcpyHostToDevice);
			paraBank = reinterpret_cast<T*>(ptr_temp);
		}
		delete[] paraBank_temp;

		bool* paraBankFlag_temp = new bool[paraBankCapacity];
		for (int i = 0; i < paraBankCapacity; ++i) { paraBankFlag_temp[i] = false; }
		{
			CUdeviceptr ptr_temp;
			size_t size_temp = sizeof(bool) * paraBankCapacity;
			cudaMalloc(reinterpret_cast<void**>(&ptr_temp), size_temp);
			cudaMemcpy(reinterpret_cast<void*>(ptr_temp), paraBankFlag_temp, size_temp, cudaMemcpyHostToDevice);
			paraBankFlag = reinterpret_cast<bool*>(ptr_temp);
		}
		delete[] paraBankFlag_temp;

		double* paraTally_temp = new double[paraTallyNum];
		for (int i = 0; i < paraTallyNum; ++i) { paraTally_temp[i] = 0; }
		{
			CUdeviceptr ptr_temp;
			size_t size_temp = sizeof(double) * paraTallyNum;
			cudaMalloc(reinterpret_cast<void**>(&ptr_temp), size_temp);
			cudaMemcpy(reinterpret_cast<void*>(ptr_temp), paraTally_temp, size_temp, cudaMemcpyHostToDevice);
			paraTally = reinterpret_cast<double*>(ptr_temp);
		}
		delete[] paraTallyDouble_temp;
	}
	__device__ void reset() {
		paraBankCount = 0;
		for (int i = 0; i < paraTallyNum; ++i) {
			paraTallyInt[i] = 0;
			paraTallyDouble[i] = 0;
		}
	}
	__device__ int count() {
		paraBankCount = 0;
		for (int i = 0; i < paraTallyNum; ++i) {
			paraBankCount += paraTallyInt[i];
		}
		if (paraBankCount == 0) {
			for (int i = 0; i < paraBankCapacity; ++i) {
				if (paraBankFlag[i] == true) {
					paraBankCount++;
				}
			}
		}
		return paraBankCount;
	}

	__device__ T sumPAT() {
		T sum_temp = 0;
		for (int i = 0; i < paraTallyNum; ++i) {
			sum_temp += paraTally[i];
		}
		return sum_temp;
	}

	__host__ __device__ int& size() {
		return paraBankCount;
	}
	__host__ __device__ int& capacity() {
		return paraBankCapacity;
	}
	__device__ T& operator[](int index) {
		return paraBank[index];
	}
	__host__ __device__ T*& ptr() {
		return paraBank;
	}
	// 数据库计数器记录
	__device__ void PABRecord(int tid, T val) {
		int rem = tid % paraTallyNum;
		int old = atomicAdd(&paraTallyInt[rem], 1);
		int index = old * paraTallyNum + rem;
		paraBank[index] = val;
		paraBankFlag[index] = true;
	}
	// 计数器记录
	__device__ void PATRecordDouble(int tid, T val) {
		int rem = tid % paraTallyNum;
		atomicAdd(paraTallyDouble+rem, val);
	}
};


double reduceBank(int sumRange, double* g_idata);

/**
 * @brief 对计数器paraRecord-paraBank结果进行归约
 */
#define Dreduce_bank(Tally, d_var, d_var_type, d_var_member, N, type) \
{ \
	ParaTally<type> tally_h; \
	cudaMemcpyFromSymbol(&tally_h, Tally, sizeof(ParaTally<type>)); \
	type TallyKeffCol_result = reduceBank(N, tally_h.paraBank); \
	cudaMemcpyToSymbol(d_var, &TallyKeffCol_result, sizeof(type), offsetof(d_var_type, d_var_member)); \
}

class CDFissBank_GPU;
void BankRedistribution_GPUh(unsigned long long neuNum, unsigned long long bankSize);


#endif // __UPDATE_GPU_H__