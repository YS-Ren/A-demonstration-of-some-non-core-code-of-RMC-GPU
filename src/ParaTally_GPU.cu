#include"ParaTally_GPU.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include "GPUConstVar.h"
#include"FissBank_GPU.h"

double reduceBank(int sumRange, double* g_idata) {
	int sum_temp = 0;
	int threadsPerBlock = 128;
	int blocksPerGrid = (sumRange + threadsPerBlock - 1) / threadsPerBlock;
	double* g_odata = nullptr;
	size_t size_temp = sizeof(double);
	cudaMalloc(reinterpret_cast<void**>(&g_odata) , size_temp);
	double h_odata = 0;
	cudaMemcpy(g_odata, &h_odata, size_temp, cudaMemcpyHostToDevice);
	reduceAdd_GPUh(g_idata, g_odata, sumRange, threadsPerBlock);
	cudaDeviceSynchronize();
	cudaMemcpy(&h_odata, g_odata, size_temp, cudaMemcpyDeviceToHost);
	cudaFree(g_odata);
	return h_odata;
}

/**
* @brief 下列函数用于对PAB裂变库的元素进行紧密重排
*/
__global__ void CollectIndex(int* d_vacant, int* d_overflow, int* d_counter_vacant, int* d_counter_overflow,
    unsigned long long neuNum, unsigned long long bankSize) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= bankSize) return;
    if (pos < neuNum) {
        if (vFissParaBankFlag_GPU[pos] == false) {
            int idx = atomicAdd(d_counter_vacant, 1);
            d_vacant[idx] = pos; // Store from index 1 onwards
        }
    }
    else {
        if (vFissParaBankFlag_GPU[pos] == true) {
            int idx = atomicAdd(d_counter_overflow, 1);
            d_overflow[idx] = pos; // Store from index 1 onwards
        }
    }
    vFissParaBankFlag_GPU[pos] = false;
}

__global__ void MoveElements(const int* d_overflow, const int* d_vacant, int threadNum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= threadNum) return;
    vFissParaSrc_GPU[d_vacant[tid]] = vFissParaSrc_GPU[d_overflow[tid]];
}

void BankRedistribution_GPUh(unsigned long long neuNum, unsigned long long bankSize) {
    int* d_vacant, * d_overflow, * d_counter1, * d_counter2;
    cudaMalloc(&d_vacant, neuNum * sizeof(int));
    cudaMalloc(&d_overflow, neuNum * sizeof(int));
    cudaMalloc(&d_counter1, sizeof(int)); cudaMemset(d_counter1, 0, sizeof(int));
    cudaMalloc(&d_counter2, sizeof(int)); cudaMemset(d_counter2, 0, sizeof(int));

    int numBlocks = (bankSize + blockSize - 1) / blockSize;
    CollectIndex << <numBlocks, blockSize >> > (d_vacant, d_overflow, d_counter1, d_counter2, neuNum, bankSize); cudaDeviceSynchronize();

    int h_counter1 = 0;
    int h_counter2 = 0;
    cudaMemcpy(&h_counter1, d_counter1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_counter2, d_counter2, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_counter1 != h_counter2) {
        printf("GPU Error: Vacant counter is not equal to overflow counter!\n");
    }
    if (h_counter1 > 0) {
        numBlocks = (h_counter1 + blockSize - 1) / blockSize;
        MoveElements << <numBlocks, blockSize >> > (d_overflow, d_vacant, h_counter1); cudaDeviceSynchronize();
    }
    else {
        printf("GPU Error: No bank neu need to be moved!\n");
    }

    cudaFree(d_vacant);
    cudaFree(d_overflow);
    cudaFree(d_counter1);
    cudaFree(d_counter2);
}