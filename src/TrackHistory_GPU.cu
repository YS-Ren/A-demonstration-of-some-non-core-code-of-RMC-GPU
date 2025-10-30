#include "NeutronTracking_GPU.h"
#include "Criticality_GPU.h"
#include "Timers_GPU.h"
#include"EventControl_GPU.h"
#include"Tally_GPU.h"

void TrackHistory_GPUh(int threadNum) {
    int numBlocks = (threadNum + blockSize - 1) / blockSize;

    int NeuNumAlive = threadNum;
    cudaMemcpyToSymbol(NeuNumAlive_GPU, &NeuNumAlive, sizeof(int));
    int zero = 0;
    int h_NeuCrossNum = threadNum;
    int h_NeuColliNum = 0;
    int iter = 0;
    RayTracking_Init_GPUg << <numBlocks, blockSize >> > (threadNum);  cudaDeviceSynchronize();
    for (;;) {
        /// - 初始化穿面粒子与碰撞粒子数
        cudaMemcpyToSymbol(ParticlesEventState_GPU, &zero, sizeof(int), offsetof(CDParticlesEventState_GPU, NeuCrossNum));
        cudaMemcpyToSymbol(ParticlesEventState_GPU, &zero, sizeof(int), offsetof(CDParticlesEventState_GPU, NeuColliNum));

        /// - 处理穿面
        if (h_NeuCrossNum != 0) {
            numBlocks = (h_NeuCrossNum + blockSize - 1) / blockSize;
            RayTracking_GPUg << <numBlocks, blockSize >> > (h_NeuCrossNum); cudaDeviceSynchronize();
        }
        /// - 处理碰撞\n
        if (h_NeuColliNum != 0) {
            numBlocks = (h_NeuColliNum + blockSize - 1) / blockSize;
            TreatColli_GPUg << <numBlocks, blockSize >> > (h_NeuColliNum);
        }
        cudaDeviceSynchronize();
        numBlocks = (threadNum + blockSize - 1) / blockSize;
        CountAliveNeuNum_GPUg << <numBlocks, blockSize >> > (threadNum);
        InterchangePSEventPtr_GPUg << <1, 1 >> > (); cudaDeviceSynchronize();
        cudaMemcpyFromSymbol(&NeuNumAlive, NeuNumAlive_GPU, sizeof(int));

        if (NeuNumAlive <= NumOfSP) {
            break;
        }

        /// - 获取穿面粒子与碰撞粒子数
        cudaMemcpyFromSymbol(&h_NeuCrossNum, ParticlesEventState_GPU, sizeof(int), offsetof(CDParticlesEventState_GPU, NeuCrossNum));
        cudaMemcpyFromSymbol(&h_NeuColliNum, ParticlesEventState_GPU, sizeof(int), offsetof(CDParticlesEventState_GPU, NeuColliNum));

        iter++;
    }
    InterchangePSEventPtr_Cross_GPUg << <1, 1 >> > (); cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(&h_NeuColliNum, ParticlesEventState_GPU, sizeof(int), offsetof(CDParticlesEventState_GPU, NeuColliNum));
    if (h_NeuColliNum != 0) {
        numBlocks = (h_NeuColliNum + blockSize - 1) / blockSize;
        TreatColli_GPUg << <numBlocks, blockSize >> > (h_NeuColliNum);
        cudaDeviceSynchronize();
    }
    InterchangePSEventPtr_Cross_GPUg << <1, 1 >> > (); cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(&NeuNumAlive, NeuNumAlive_GPU, sizeof(int));
    cudaMemcpyFromSymbol(&h_NeuCrossNum, ParticlesEventState_GPU, sizeof(int), offsetof(CDParticlesEventState_GPU, NeuCrossNum));
    cudaMemcpyToSymbol(ParticlesEventState_GPU, &zero, sizeof(int), offsetof(CDParticlesEventState_GPU, NeuCrossNum));
    cudaMemcpyToSymbol(ParticlesEventState_GPU, &zero, sizeof(int), offsetof(CDParticlesEventState_GPU, NeuColliNum));
    if (h_NeuCrossNum != 0) {
        numBlocks = (h_NeuCrossNum + blockSize - 1) / blockSize;
        TrackHistoryAfterEvent_GPUg << <numBlocks, blockSize >> > (h_NeuCrossNum);
        cudaDeviceSynchronize();
    }
}
