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
// С�����ڴ�أ���������ͷ�
class CDGPUMemPool {
public:
	// ����
	void expansion();
	// ��ʼ��
	void initialize(size_t size = 16384);
	// ���캯��
	CDGPUMemPool(size_t size = 16384) {
		initialize(size);
	}
	// �ӳ��з����ڴ�
	void* malloc(size_t size);
	// �ͷſռ�
	void freeAll();
	// ��ӡ��Ϣ
	void print();
	// �ж�����ָ���Ƿ�λ�ڳ���
	bool isInPool(void* ptr);

private:
	std::vector<void*> _GPUPoolPtr; // �̳߳���ʼָ��
	std::vector<size_t> _occupiedSpace; // ÿ���̳߳ص��ѱ�ռ�ÿռ�
	std::vector<size_t> _freeSpace; // ÿ���̳߳ص�ʣ��ռ�
	size_t _containerSize; // ÿ�����ݵĳߴ�
};

// GPU�ڴ洦����
class CDGPUMemory {
public:
	// ���캯��
	CDGPUMemory(size_t thresh = 256);
	// ��ʼ��
	void initialize(size_t thresh = 256);
	// �ͷſռ�
	void free(size_t index);
	void free(void* ptr);
	void freeAll();
	// ��ӡ������Ϣ
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
	std::vector<void*> _GPUPtr; // �洢�����ѷ����ڴ���豸��ָ��
	size_t _segThresh; // ������ڴ�ߴ������ֵʱ������ʹ��cudaMalloc��������ֵ��ӳ��з��䡣
	CDGPUMemPool _GPUMemPool; // С�ߴ�������ڴ�ض���
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