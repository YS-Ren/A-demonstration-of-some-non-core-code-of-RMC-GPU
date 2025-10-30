#ifndef __GPU_CONTAINER_H__
#define  __GPU_CONTAINER_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>

#include "GPUErrorCtrl.h"
#include "GPUMemory.h"
#include "GPUException.h"

//**************************************************
//	class def
//**************************************************

/**
 * @brief 自定义设备端vector
 */
template <typename T>
class dev_vector {
public:
    // 构造器
    dev_vector() = default;
    dev_vector(size_t size) {
        initialize(size);
    }
    dev_vector(dev_vector<T>& other) {
        copyFrom(other);
    }
    dev_vector(std::vector<T>& vec) {
        initialize(vec);
    }

    // 初始化，size为元素数量
    void initialize(size_t size) {
        CUDA_MALLOC(_ptr, size * sizeof(T));
        _size = size;
    }
    void initialize(std::vector<T>& vec) {
        initialize(vec.size());
        {
            T* ptr_temp = new T[_size];
            for (int i = 0; i < _size; ++i) {
                ptr_temp[i] = vec[i];
            }
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(_ptr), vec.data(), _size * sizeof(T), cudaMemcpyHostToDevice));
            delete[] ptr_temp;
        }
    }
    void initialize(std::vector<std::vector<T>>& other) {
        copyFrom2D(other);
    }
    // 释放空间
    void free() {
        CUDA_FREE(_ptr);
        _ptr = nullptr;
        _size = 0;
    }
    // 元素加
    void add(T val) {
        std::vector<T> h_vec(_size);
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(h_vec.data()), _ptr, _size * sizeof(T), cudaMemcpyDeviceToHost));
        for (auto& x : h_vec) {
            x += val;
        }
        CUDA_CHECK(cudaMemcpy( _ptr, reinterpret_cast<void*>(h_vec.data()), _size * sizeof(T), cudaMemcpyHostToDevice));
    }
    // 元素乘
    void mul(T val) {
        std::vector<T> h_vec(_size);
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(h_vec.data()), _ptr, _size * sizeof(T), cudaMemcpyDeviceToHost));
        for (auto& x : h_vec) {
            x *= val;
        }
        CUDA_CHECK(cudaMemcpy(_ptr, reinterpret_cast<void*>(h_vec.data()), _size * sizeof(T), cudaMemcpyHostToDevice));
    }
    // 访问器
    __host__ __device__ T& operator[](size_t index) {
#ifdef __CUDA_ARCH__
        if (index < _size) {
            return _ptr[index];
        }
        else {
            size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
            char log[64];
            indexInfoToString(index, _size, log);
            RECORD_ERROR(eGPUErrorType::vecOverRange, tid, log);
            return _ptr[0];
        }
#else
        std::stringstream ss;
        ss << "The operator[] is only used in device!" << __POSITION___;
        OLogger.error(ss.str());
        throw Exception(ss.str().c_str());
#endif
    }
    __host__ __device__ const T& operator[](size_t index) const {
#ifdef __CUDA_ARCH__
        if (index < _size) {
            return _ptr[index];
        }
        else {
            size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
            char log[64];
            indexInfoToString(index, _size, log);
            RECORD_ERROR(eGPUErrorType::vecOverRange, tid, log);
            return _ptr[0];
        }
#else
        std::stringstream ss;
        ss << "The operator[] is only used in device!" << __POSITION___;
        OLogger.error(ss.str());
        throw Exception(ss.str().c_str());
#endif
    }
    __host__ __device__ T*& ptr() {
        return _ptr;
    }
    __host__ __device__ T* ptr(size_t index) {
#ifdef __CUDA_ARCH__
        if (index < _size) {
            return _ptr + index;
        }
        else {
            size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
            char log[64];
            indexInfoToString(index, _size, log);
            RECORD_ERROR(eGPUErrorType::vecOverRange, tid, log);
            return _ptr;
        }
#else
        std::stringstream ss;
        ss << "The operator ptr(index) is only used in device!" << __POSITION___;
        OLogger.error(ss.str());
        throw Exception(ss.str().c_str());
        exit(-1);
#endif
    }
    __host__ __device__ T& back() {
        return _ptr[_size - 1];
    }
    __host__ __device__ size_t& size() {
        return _size;
    }
    __host__ __device__ bool empty() {
        return (_size == 0);
    }
    __host__ __device__ T* begin() {
        return _ptr;
    }
    __host__ __device__ T* end() {
        return _ptr + _size;
    }
    __host__ __device__ bool operator==(dev_vector<T>& other) {
        if (_size != other.size()) {
            return false;
        }
        for (int i = 0; i < _size; ++i) {
            if (_ptr[i] != other[i]) {
                return false;
            }
        }
        return true;
    }
    __host__ void copyTo(dev_vector<T>& other) {
        if (other.size() != 0) {
            other.free();
        }
        other.initialize(_size);
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(other.ptr()), reinterpret_cast<void*>(_ptr),
            _size * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    __host__ void copyTo(std::vector<T>& other) {
        other.resize(_size);
        CUDA_CHECK(cudaMemcpy(other.data(), reinterpret_cast<void*>(_ptr), _size * sizeof(T), cudaMemcpyDeviceToHost));
    }
    __host__ void copyFrom(dev_vector<T>& other) {
        if (other.size() == 0) {
            std::stringstream ss;
            ss << "The input dev_vector is empty!" << __POSITION___;
            OLogger.error(ss.str());
            throw Exception(ss.str().c_str());
        }
        if (_size != 0) {
            free();
        }
        initialize(other.size());
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(_ptr), reinterpret_cast<void*>(other.ptr()),
            _size * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    __host__ __device__ void interchange(dev_vector<T>& other) {
        // interchange ptr
        T* ptr_temp;
        ptr_temp = this->_ptr;
        this->_ptr = other._ptr;
        other._ptr = ptr_temp;
        // interchange size
        int size_temp;
        size_temp = this->_size;
        this->_size = other._size;
        other._size = size_temp;
    }
private:
    T* _ptr{ nullptr };
    size_t _size{ 0 };
}; // dev_vector

template <typename T>
void VecC2G(std::vector<T>& src, dev_vector<T>& dst) {
    dst.initialize(src);
}
template <typename T>
void VecC2G(std::vector<std::vector<T>>& src, dev_vector<dev_vector<T>>& dst) {
    std::vector<dev_vector<T>> vec2D_temp;
    for (int i = 0; i < src.size(); ++i) {
        vec2D_temp.emplace_back(src[i]);
    }
    dst.initialize(vec2D_temp);
}

template <typename T>
void VecG2C(dev_vector<T>& src, std::vector<T>& dst) {
    src.copyTo(dst);
}
template <typename T>
void VecG2C(dev_vector<dev_vector<T>>& src,  std::vector<std::vector<T>>& dst) {
    dst.resize(0);
    std::vector<dev_vector<T>> vec2D_temp;
    src.copyTo(vec2D_temp);
    for (int i = 0; i < vec2D_temp.size(); ++i) {
        std::vector<T> vec_temp;
        vec2D_temp[i].copyTo(vec_temp);
        dst.push_back(vec_temp);
    }
}

template <typename T>
void VecFree(dev_vector<T>& vec) {
    vec.free();
}
template <typename T>
void VecFree(dev_vector<dev_vector<T>>& vec) {
    std::vector<dev_vector<T>> h_vec;
    VecG2C(vec, h_vec);
    for (auto& v : h_vec) {
        v.free();
    }
    vec.free();
}

#endif //__GPU_CONTAINER_H__