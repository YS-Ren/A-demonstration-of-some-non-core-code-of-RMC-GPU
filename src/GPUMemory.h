#ifndef __GPU_MEMORY_H__
#define  __GPU_MEMORY_H__

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include"GPUException.h"

//**************************************************
//	class def
//**************************************************
// 小变量内存池，分配后不再释放
class CDGPUMemPool {
public:
	// 扩容
	void expansion();
	// 初始化
	void initialize(size_t size = 16384);
	// 构造函数
	CDGPUMemPool(size_t size = 16384) {
		initialize(size);
	}
	// 从池中分配内存
	void* malloc(size_t size);
	// 释放空间
	void freeAll();
	// 打印信息
	void print();
	// 判断输入指针是否位于池中
	bool isInPool(void* ptr);

private:
	std::vector<void*> _GPUPoolPtr; // 线程池起始指针
	std::vector<size_t> _occupiedSpace; // 每个线程池的已被占用空间
	std::vector<size_t> _freeSpace; // 每个线程池的剩余空间
	size_t _containerSize; // 每次扩容的尺寸
};

// GPU内存处理类
class CDGPUMemory {
public:
	// 构造函数
	CDGPUMemory(size_t thresh = 256);
	// 初始化
	void initialize(size_t thresh = 256);
	// 释放空间
	void free(size_t index);
	void free(void* ptr);
	void freeAll();
	// 打印调试信息
	void print();

	template<typename T>
	void malloc(T*& devPtr, size_t size) {
		void* ptr_temp = reinterpret_cast<void*>(devPtr);
		if (size < _segThresh) {
			ptr_temp = _GPUMemPool.malloc(size);
		}
		else {
			CUDA_CHECK(cudaMalloc(&ptr_temp, size));
			auto it = std::find(_GPUPtr.begin(), _GPUPtr.end(), nullptr);
			if (it != _GPUPtr.end()) {
				*it = ptr_temp; 
			}
			else {
				_GPUPtr.push_back(ptr_temp); 
			}
		}
		devPtr = reinterpret_cast<T*>(ptr_temp);
	}

private:
	std::vector<void*> _GPUPtr; // 存储所有已分配内存的设备端指针
	size_t _segThresh; // 分配的内存尺寸高于阈值时，正常使用cudaMalloc，低于阈值则从池中分配。
	CDGPUMemPool _GPUMemPool; // 小尺寸变量的内存池对象
};

// print GPU 0 memory information
void GPUPrintMemory();

//**************************************************
//	extern var
//**************************************************

extern CDGPUMemory OGPUMemory;

#define CUDA_MALLOC(ptr, size) OGPUMemory.malloc(ptr, size);
#define CUDA_FREE(ptr) OGPUMemory.free(ptr);



#endif //__GPU_MEMORY_H__