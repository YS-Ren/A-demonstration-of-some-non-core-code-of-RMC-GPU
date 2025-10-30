#include "GPUMemory.h"

#include "GPUPrint.h"
#include "GPUException.h"

//**************************************************
//	class def
//**************************************************

// 小变量内存池，分配后不再释放
// class CDGPUMemPool
void CDGPUMemPool::expansion() {
	void* ptr;
	CUDA_CHECK(cudaMalloc(&ptr, _containerSize));
	_GPUPoolPtr.push_back(ptr);
	_occupiedSpace.push_back(0);
	_freeSpace.push_back(_containerSize);
}

void CDGPUMemPool::initialize(size_t size) {
	_containerSize = size;
	expansion();
}

void* CDGPUMemPool::malloc(size_t size) {
	void* new_ptr = nullptr;
	for (int i = 0; i < _GPUPoolPtr.size(); ++i) {
		if (_freeSpace[i] >= size) {
			new_ptr = static_cast<char*>(_GPUPoolPtr[i]) + _occupiedSpace[i];
			_freeSpace[i] -= size;
			_occupiedSpace[i] += size;
			return new_ptr;
		}
	}
	expansion();
	new_ptr = _GPUPoolPtr.back();
	_freeSpace.back() -= size;
	_occupiedSpace.back() += size;
	return new_ptr;
}

void CDGPUMemPool::freeAll() {
	for (auto& ptr : _GPUPoolPtr) {
		CUDA_CHECK(cudaFree(ptr));
	}
}

void CDGPUMemPool::print() {
	{
		std::stringstream ss;
		ss << "Info" << __POSITION___;
		OLogger.header(ss.str(), "CDGPUMemPool", CDLogger::DEBUG);
	}
	{
		std::stringstream ss;
		ss << "GPUMemPool: num = " << _GPUPoolPtr.size() << ", _freeSpace = " << _containerSize;
		OLogger.main(ss.str());
	}
	for (int i = 0; i < _GPUPoolPtr.size(); ++i) {
		std::stringstream ss;
		ss << "[" << i << "] occpiedSpace = " << _occupiedSpace[i] << ", _freeSpace = " << _freeSpace[i];
		OLogger.main(ss.str());
	}
}

bool CDGPUMemPool::isInPool(void* ptr) {
	bool flag = false;
	for (int i = 0; i < _GPUPoolPtr.size(); ++i) {
		const char* p = static_cast<const char*>(ptr);
		const char* start = static_cast<const char*>(_GPUPoolPtr[i]);
		const char* end = static_cast<const char*>(_GPUPoolPtr[i]) + _occupiedSpace[i];
		if ((p >= start) && (p < end)) {
			flag = true;
		}
	}
	return flag;
}

// GPU内存处理类
//class CDGPUMemory 
CDGPUMemory::CDGPUMemory(size_t thresh) {
		initialize(thresh);
	}

void CDGPUMemory::initialize(size_t thresh) {
	_segThresh = thresh;
	_GPUMemPool.initialize();
}

void CDGPUMemory::free(size_t index) {
	if (index >= _GPUPtr.size()) {
		std::stringstream ss;
		ss << "The index waiting to free is over range in _GPUPtr!" << __POSITION___;
		OLogger.error(ss.str());
		throw Exception(ss.str().c_str());
	}
	if (_GPUPtr[index] != nullptr) {
		CUDA_CHECK(cudaFree(_GPUPtr[index]));
		_GPUPtr[index] = nullptr;
	}
}

void CDGPUMemory::free(void* ptr) {
	if (_GPUMemPool.isInPool(ptr)) {
		return;
	}
	auto it = std::find(_GPUPtr.begin(), _GPUPtr.end(), ptr);
	if (it != _GPUPtr.end()) {
		*it = nullptr;
		CUDA_CHECK(cudaFree(ptr));
	}
	{
		std::stringstream ss;
		ss << "The ptr waiting to free does not exist!" << __POSITION___;
		OLogger.error(ss.str());
		throw Exception(ss.str().c_str());
	}
}

void CDGPUMemory::freeAll() {
	for (size_t i = 0; i < _GPUPtr.size(); ++i) {
		free(i);
	}
	_GPUMemPool.freeAll();
}

void CDGPUMemory::print() {
	size_t occpCount = 0;
	for (void* ptr : _GPUPtr) {
		occpCount += (ptr != nullptr);
	}
	{
		std::stringstream ss;
		ss << "Info" << __POSITION___;
		OLogger.header(ss.str(), "CDGPUMemory", CDLogger::DEBUG);
	}
	{
		std::stringstream ss;
		ss << "_GPUPtr.size = " << _GPUPtr.size() << ", _GPUPtr.occp = " << occpCount<<", _segThresh = "<< _segThresh;
		OLogger.main(ss.str());
	}
	_GPUMemPool.print();
}

// print GPU 0 memory information
void GPUPrintMemory() {
	cudaSetDevice(0);
	size_t freeMem, totalMem;
	cudaMemGetInfo(&freeMem, &totalMem);
	freeMem /= 1048576;
	totalMem /= 1048576;

	std::stringstream ss;
	ss << "GPU 0 memory info" << __POSITION___;
	OLogger.header(ss.str(), "GPU Memory");

	CDTable table;
	table.addColumn("Memory", eAlignType::LEFT);
	table.addColumn("Value", eAlignType::LEFT);
	table.addColumn("Unit", eAlignType::LEFT);

	table.addRow({ "Total", std::to_string(totalMem), "MB" });
	table.addRow({ "Free", std::to_string(freeMem), "MB" });
	table.addRow({ "Used", std::to_string(totalMem - freeMem), "MB" });
	table.print();
}

//**************************************************
//	extern var
//**************************************************
CDGPUMemory OGPUMemory;
