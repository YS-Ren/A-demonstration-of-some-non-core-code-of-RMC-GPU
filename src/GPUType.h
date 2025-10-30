#ifndef __GPU_TYPE_H__
#define __GPU_TYPE_H__

#include "RMC_GPU.h"

/**
 * @brief GPU错误检查宏
 */
#define CudaCheck(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

 /**
  * @brief 核函数错误检查宏
  */
#define CudaCheckGlobal CudaCheck(cudaGetLastError());

/**
 * @brief 用于定义设备端dev_vector变量的宏组
 */
#define Dextern_vector(d_var,type) extern __device__ dev_vector<type> d_var;
#define Ddecl_vector(d_var,type) __device__ dev_vector<type> d_var;
 //#define Dtrans_varArray(h_var,d_var) cudaMemcpyToSymbol(d_var, h_var, sizeof(decltype(h_var)));

#define TdevVar(h_var,varname) h_var##_temp.varname=h_var.varname;

/**
 * @brief 将char[]类型的变量的数据迁移至h_var_temp变量内。注：对象变量本身未迁移，需跟随上层类变量一起迁移
 * @param h_var 主机端的char[]变量
 * @param h_var_temp 数据迁移进的char[]临时变量的名称
 * @param size char[]数组的长度
 */
#define Dtrans_cChar(h_var,h_var_temp,size) \
    for (int i_Dtrans_cChar = 0; i_Dtrans_cChar < size; ++i_Dtrans_cChar) { \
        h_var_temp[i_Dtrans_cChar] = h_var[i_Dtrans_cChar]; \
    }
/**
* @brief 将char[][]类型的变量的数据迁移至h_var_temp变量内。注：对象变量本身未迁移，需跟随上层类变量一起迁移
* @param h_var 主机端的char[][]变量
* @param h_var_temp 数据迁移进的char[][]临时变量的名称
* @param size1 char[size1][size2]数组长度
* @param size2 char[size1][size2]数组长度
*/
#define Dtrans_cChar2(h_var,h_var_temp,size1,size2) \
    for (int i = 0; i < size1; ++i) { \
        for(int j = 0; j < size2; ++j) { \
            h_var_temp[i][j] = h_var[i][j]; \
        } \
    }
/**
* @brief 用于类内，将char[]类型的成员变量的数据迁移同名_temp变量内。注：对象变量本身未迁移，需跟随上层类变量一起迁移
* @param h_var 主机端的类名称
* @param varname 类内char[]变量的名称
* @param size char[]数组的长度
*/
#define Tchar(h_var,varname,size) \
    for (int i_Tchar = 0; i_Tchar < size; ++i_Tchar) { \
        h_var##_temp.varname[i_Tchar] = h_var.varname[i_Tchar]; \
    }
/**
* @brief 用于类内，将char*类型的成员变量指向数据迁移到设备端，并将设备端指针赋予同名_temp变量内。注：对象变量本身未迁移，需跟随上层类变量一起迁移
* @param h_var 主机端的类名称
* @param varname 类内char*变量的名称
*/
#define Dtrans_devCharPtr(h_var,varname) \
{ \
    CUdeviceptr ptr; \
    const size_t size = sizeof(char) * strlen(h_var.varname); \
    cudaMalloc(reinterpret_cast<void**>(&ptr), size); \
    CudaCheck(cudaMemcpy( \
        reinterpret_cast<void*>(ptr), \
        h_var.varname, \
        size, \
        cudaMemcpyHostToDevice)); \
    h_var##_temp.varname = reinterpret_cast<char*>(ptr); \
}

/**
* @brief 用于类内，将std::string类型的成员变量指向数据迁移到设备端，并将设备端指针赋予同名_temp变量内。注：对象变量本身未迁移，需跟随上层类变量一起迁移
* @param h_var 主机端的std::string变量的名称
* @param h_var_temp 类内std::string临时变量的名称
*/
#define Dtrans_cString(h_var,h_var_temp) \
{ \
    CUdeviceptr ptr; \
    size_t size = sizeof(char) * (h_var.size() + 1); \
    cudaMalloc(reinterpret_cast<void**>(&ptr), size); \
    cudaMemcpy(\
        reinterpret_cast<void*>(ptr), \
        h_var.c_str(), \
        size, \
        cudaMemcpyHostToDevice); \
    h_var_temp = reinterpret_cast<char*>(ptr); \
    h_var_temp##_size = h_var.size(); \
}

/**
* @brief 用于类内，将数据迁移到自定义设备端vector的宏。注：dev_vector对象变量本身未迁移，需跟随上层类变量一起迁移
* @param h_var主机端的类内vectr
* @param h_var_temp用设备端的类定义的，用于迁移数据的主机端变量
*/
#define Dtrans_devVec(h_var,h_var_temp) \
    { \
        CUdeviceptr ptr; \
        const size_t size = sizeof(decltype(h_var)::value_type)*h_var.size(); \
        cudaMalloc(reinterpret_cast<void**>(&ptr), size); \
        CudaCheck(cudaMemcpy( \
        reinterpret_cast<void*>(ptr), \
        h_var.data(), \
        size, \
        cudaMemcpyHostToDevice)); \
        h_var_temp.ptr()=reinterpret_cast<decltype(h_var)::value_type*>(ptr); \
        h_var_temp.size()= h_var.size(); \
    }
#define TdevVec(h_var,varname) Dtrans_devVec(h_var.varname,h_var##_temp.varname)

/**
* @brief 用于类内，将数据迁移到自定义设备端【二维】vector 的宏。注：dev_vector对象变量本身未迁移，需跟随上层类变量一起迁移
* @param h_var主机端的类内【二维】vectr
* @param h_var_temp用设备端的类定义的，用于迁移数据的主机端变量
*/
#define Dtrans_devVec2(h_var,h_var_temp) \
    { \
        std::vector<dev_vector<decltype(h_var)::value_type::value_type>> dev_vec_ptr_temp; \
        for (int Dtrans_decVec2_i = 0; Dtrans_decVec2_i < h_var.size(); ++Dtrans_decVec2_i) { \
            dev_vector<decltype(h_var)::value_type::value_type> dev_vec_temp2; \
            { \
                CUdeviceptr ptr; \
                const size_t size = sizeof(decltype(h_var)::value_type::value_type) * h_var[Dtrans_decVec2_i].size(); \
                cudaMalloc(reinterpret_cast<void**>(&ptr), size); \
                CudaCheck(cudaMemcpy(reinterpret_cast<void*>(ptr), h_var[Dtrans_decVec2_i].data(), size, cudaMemcpyHostToDevice)); \
                dev_vec_temp2.ptr() = reinterpret_cast<decltype(h_var)::value_type::value_type*>(ptr); \
                dev_vec_temp2.size() = h_var[Dtrans_decVec2_i].size(); \
            }; \
            dev_vec_ptr_temp.push_back(dev_vec_temp2); \
        } \
        CUdeviceptr Ptr; \
        const size_t Size = sizeof(dev_vector<decltype(h_var)::value_type::value_type>) * h_var.size(); \
        cudaMalloc(reinterpret_cast<void**>(&Ptr), Size); \
        CudaCheck(cudaMemcpy(reinterpret_cast<void*>(Ptr), dev_vec_ptr_temp.data(), Size, cudaMemcpyHostToDevice)); \
        h_var_temp.ptr() = reinterpret_cast<dev_vector< decltype(h_var)::value_type::value_type >*>(Ptr); \
        h_var_temp.size() = h_var.size(); \
    }
#define TdevVec2(h_var,varname) Dtrans_devVec2(h_var.varname,h_var##_temp.varname)

/**
* @brief 用于类内，将数据迁移到自定义设备端【三维】vector 的宏。注：dev_vector对象变量本身未迁移，需跟随上层类变量一起迁移
* @param h_var主机端的类内【三维】vectr
* @param h_var_temp用设备端的类定义的，用于迁移数据的主机端变量
*/
#define Dtrans_devVec3(h_var,h_var_temp) \
   { \
        std::vector< dev_vector<dev_vector<decltype(h_var)::value_type::value_type::value_type>>> h_vec_temp1; \
        for (int Dtrans_decVec3_j = 0; Dtrans_decVec3_j < h_var.size(); ++Dtrans_decVec3_j) { \
            dev_vector<dev_vector<decltype(h_var)::value_type::value_type::value_type>> dev_vec_temp2; \
            std::vector<dev_vector<decltype(h_var)::value_type::value_type::value_type>> h_vec_temp2; \
            for (int Dtrans_decVec3_i = 0; Dtrans_decVec3_i < h_var[Dtrans_decVec3_j].size(); ++Dtrans_decVec3_i) { \
                dev_vector<decltype(h_var)::value_type::value_type::value_type> dev_vec_temp3; \
                { \
                    CUdeviceptr ptr3; \
                    const size_t size1 = sizeof(decltype(h_var)::value_type::value_type::value_type) * h_var[Dtrans_decVec3_j][Dtrans_decVec3_i].size(); \
                    cudaMalloc(reinterpret_cast<void**>(&ptr3), size1); \
                    CudaCheck(cudaMemcpy(reinterpret_cast<void*>(ptr3), h_var[Dtrans_decVec3_j][Dtrans_decVec3_i].data(), size1, cudaMemcpyHostToDevice)); \
                    dev_vec_temp3.ptr() = reinterpret_cast<decltype(h_var)::value_type::value_type::value_type*>(ptr3); \
                    dev_vec_temp3.size() = h_var[Dtrans_decVec3_j][Dtrans_decVec3_i].size(); \
                }; \
                h_vec_temp2.push_back(dev_vec_temp3); \
            } \
            { \
                CUdeviceptr ptr2; \
                const size_t size2 = sizeof(dev_vector<decltype(h_var)::value_type::value_type::value_type>) * h_var[Dtrans_decVec3_j].size(); \
                cudaMalloc(reinterpret_cast<void**>(&ptr2), size2); \
                CudaCheck(cudaMemcpy(reinterpret_cast<void*>(ptr2), h_vec_temp2.data(), size2, cudaMemcpyHostToDevice)); \
                dev_vec_temp2.ptr() = reinterpret_cast<dev_vector<decltype(h_var)::value_type::value_type::value_type>*>(ptr2); \
                dev_vec_temp2.size() = h_var[Dtrans_decVec3_j].size(); \
            } \
            h_vec_temp1.push_back(dev_vec_temp2); \
        } \
        { \
            CUdeviceptr ptr1; \
            const size_t size1 = sizeof(dev_vector<dev_vector<decltype(h_var)::value_type::value_type::value_type>>) * h_var.size(); \
            cudaMalloc(reinterpret_cast<void**>(&ptr1), size1); \
            CudaCheck(cudaMemcpy(reinterpret_cast<void*>(ptr1), h_vec_temp1.data(), size1, cudaMemcpyHostToDevice)); \
            h_var_temp.ptr() = reinterpret_cast<dev_vector<dev_vector<decltype(h_var)::value_type::value_type::value_type>>*>(ptr1); \
            h_var_temp.size() = h_var.size(); \
        } \
   }
#define TdevVec3(h_var,varname) Dtrans_devVec3(h_var.varname,h_var##_temp.varname)

/**
* @brief 用于类内，将类向量vector<ClassType>迁移到自定义设备端dev_vector<ClassType_GPU>的宏。注：dev_vector对象变量本身未迁移，需跟随上层类变量一起迁移
* @param h_var主机端的类向量vectr<ClassType>
* @param h_var_temp用设备端的dev_vectr<ClassType_GPU>定义的，用于迁移数据的主机端变量
* @param type 主机端类的名称ClassType
*/
#define Dtrans_devClassVec(h_var,h_var_temp,type) \
    { \
    type##_GPU* h_var_ptr = new  type##_GPU[h_var.size()]; \
    for (int i = 0; i < h_var.size(); ++i) { \
            h_var_ptr[i] = h_var[i]; \
    } \
    { \
    CUdeviceptr ptr_temp; \
    const size_t size = sizeof(type##_GPU) * h_var.size(); \
    cudaMalloc(reinterpret_cast<void**>(&ptr_temp), size); \
    CudaCheck(cudaMemcpy(\
        reinterpret_cast<void*>(ptr_temp), \
        h_var_ptr, \
        size, \
        cudaMemcpyHostToDevice)); \
            h_var_temp.size() = h_var.size(); \
            h_var_temp.ptr() = reinterpret_cast<type##_GPU*>(ptr_temp); \
    } \
    delete[] h_var_ptr; \
    }

#define Dtrans_ClassVec_t(h_var,h_var_temp) \
    for (int i = 0; i < h_var.size(); ++i) { \
            h_var_temp.push_back({}); \
            h_var_temp[i] = h_var[i]; \
    } 

#define Dtrans_ClassVec2_t(h_var,h_var_temp) \
    for (int i = 0; i < h_var.size(); ++i) { \
        h_var_temp.push_back({}); \
        for(int j=0;j< h_var[i].size();++j){ \
            h_var_temp[i].push_back({}); \
            h_var_temp[i][j] = h_var[i][j]; \
        } \
    } 

#define Dtrans_ClassVec3_t(h_var,h_var_temp) \
    for (int i = 0; i < h_var.size(); ++i) { \
        h_var_temp.push_back({}); \
        for(int j=0;j< h_var[i].size();++j){ \
            h_var_temp[i].push_back({}); \
            for(int k=0;k<h_var[i][j].size();++k){ \
                h_var_temp[i][j].push_back({}); \
                h_var_temp[i][j][k] = h_var[i][j][k]; \
            } \
        } \
    } 
/**
* @brief 用于类内，将类向量vector<vector<ClassType>>迁移到自定义设备端dev_vector<dev_vector<ClassType_GPU>>的宏。注：dev_vector对象变量本身未迁移，需跟随上层类变量一起迁移
* @param h_var主机端的类向量vector<vector<ClassType>>
* @param h_var_temp用设备端的dev_vector<dev_vector<ClassType_GPU>>定义的，用于迁移数据的主机端变量
* @param type 主机端类的名称ClassType
*/
#define Dtrans_devClassVec2(h_var,h_var_temp,type) \
    { \
        std::vector<dev_vector<type##_GPU>> dev_vec_ptr_temp; \
        for (int Dtrans_decVec2_i = 0; Dtrans_decVec2_i < h_var.size(); ++Dtrans_decVec2_i) \
        { \
            dev_vector<type##_GPU> dev_vec_temp2; \
            type##_GPU* h_var_ptr2 = new type##_GPU[h_var[Dtrans_decVec2_i].size()]; \
            for (int j = 0; j < h_var[Dtrans_decVec2_i].size(); ++j) { \
                h_var_ptr2[j] = h_var[Dtrans_decVec2_i][j]; \
            } \
            { \
                CUdeviceptr ptr; \
                const size_t size = sizeof(type##_GPU) * h_var[Dtrans_decVec2_i].size(); \
                cudaMalloc(reinterpret_cast<void**>(&ptr), size); \
                cudaMemcpy(reinterpret_cast<void*>(ptr), h_var_ptr2, size, cudaMemcpyHostToDevice); \
                dev_vec_temp2.ptr() = reinterpret_cast<type##_GPU*>(ptr); \
                dev_vec_temp2.size() = h_var[Dtrans_decVec2_i].size(); \
            }; \
            dev_vec_ptr_temp.push_back(dev_vec_temp2); \
            delete[] h_var_ptr2; \
        } \
        CUdeviceptr Ptr; \
        const size_t Size = sizeof(dev_vector<type##_GPU>) * h_var.size(); \
        cudaMalloc(reinterpret_cast<void**>(&Ptr), Size); \
        CudaCheck(cudaMemcpy(reinterpret_cast<void*>(Ptr), dev_vec_ptr_temp.data(), Size, cudaMemcpyHostToDevice)); \
        h_var_temp.ptr() = reinterpret_cast<dev_vector< type##_GPU >*>(Ptr); \
        h_var_temp.size() = h_var.size(); \
    }

/**
* @brief 用于类内，将类向量vector<vector<vector<ClassType>>>迁移到自定义设备端dev_vector<dev_vector<dev_vector<ClassType_GPU>>>的宏。注：dev_vector对象变量本身未迁移，需跟随上层类变量一起迁移
* @param h_var主机端的类向量vector<vector<vector<ClassType>>>
* @param h_var_temp用设备端的dev_vector<dev_vector<dev_vector<ClassType_GPU>>>定义的，用于迁移数据的主机端变量
* @param type 主机端类的名称ClassType
*/
#define Dtrans_devClassVec3(h_var, h_var_temp, type) \
    { \
        std::vector< dev_vector<dev_vector<type##_GPU>>> h_vec_temp1; \
        for (int Dtrans_decVec3_j = 0; Dtrans_decVec3_j < h_var.size(); ++Dtrans_decVec3_j) { \
            dev_vector<dev_vector<type##_GPU>> dev_vec_temp2; \
            std::vector<dev_vector<type##_GPU>> h_vec_temp2; \
            for (int Dtrans_decVec3_i = 0; Dtrans_decVec3_i < h_var[Dtrans_decVec3_j].size(); ++Dtrans_decVec3_i) { \
                dev_vector<type##_GPU> dev_vec_temp3; \
                type##_GPU* h_var_ptr3 = new type##_GPU[h_var[Dtrans_decVec3_j][Dtrans_decVec3_i].size()]; \
                for (int k = 0; k < h_var[Dtrans_decVec3_j][Dtrans_decVec3_i].size(); ++k) { \
                    h_var_ptr3[k] = h_var[Dtrans_decVec3_j][Dtrans_decVec3_i][k]; \
                } \
                { \
                    CUdeviceptr ptr3; \
                    const size_t size1 = sizeof(type##_GPU) * h_var[Dtrans_decVec3_j][Dtrans_decVec3_i].size(); \
                    cudaMalloc(reinterpret_cast<void**>(&ptr3), size1); \
                    CudaCheck(cudaMemcpy(reinterpret_cast<void*>(ptr3), h_var_ptr3, size1, cudaMemcpyHostToDevice)); \
                    dev_vec_temp3.ptr() = reinterpret_cast<type##_GPU*>(ptr3); \
                    dev_vec_temp3.size() = h_var[Dtrans_decVec3_j][Dtrans_decVec3_i].size(); \
                }; \
                h_vec_temp2.push_back(dev_vec_temp3); \
                delete[] h_var_ptr3; \
            } \
            { \
                CUdeviceptr ptr2; \
                const size_t size2 = sizeof(dev_vector<type##_GPU>) * h_var[Dtrans_decVec3_j].size(); \
                cudaMalloc(reinterpret_cast<void**>(&ptr2), size2); \
                CudaCheck(cudaMemcpy(reinterpret_cast<void*>(ptr2), h_vec_temp2.data(), size2, cudaMemcpyHostToDevice)); \
                dev_vec_temp2.ptr() = reinterpret_cast<dev_vector<type##_GPU>*>(ptr2); \
                dev_vec_temp2.size() = h_var[Dtrans_decVec3_j].size(); \
            } \
            h_vec_temp1.push_back(dev_vec_temp2); \
        } \
        { \
            CUdeviceptr ptr1; \
            const size_t size1 = sizeof(dev_vector<dev_vector<type##_GPU>>) * h_var.size(); \
            cudaMalloc(reinterpret_cast<void**>(&ptr1), size1); \
            CudaCheck(cudaMemcpy(reinterpret_cast<void*>(ptr1), h_vec_temp1.data(), size1, cudaMemcpyHostToDevice)); \
            h_var_temp.ptr() = reinterpret_cast<dev_vector<dev_vector<type##_GPU>>*>(ptr1); \
            h_var_temp.size() = h_var.size(); \
        } \
    }


/**
* @brief 用于类内，将类向量vector<unique_ptr<ClassType>>迁移到自定义设备端dev_vector<ClassType_GPU>的宏。注：dev_vector对象变量本身未迁移，需跟随上层类变量一起迁移
* @param h_var主机端的类向量vector<unique_ptr<ClassType>>
* @param h_var_temp用设备端的dev_vectr<ClassType_GPU>定义的，用于迁移数据的主机端变量
* @param type 主机端类的名称ClassType
*/
#define Dtrans_devClassPtrVec(h_var,h_var_temp,type) \
    { \
    type##_GPU* h_var_ptr = new  type##_GPU[h_var.size()]; \
    for (int i = 0; i < h_var.size(); ++i) { \
            h_var_ptr[i] = *h_var[i]; \
    } \
    { \
    CUdeviceptr ptr_temp; \
    const size_t size = sizeof(type##_GPU) * h_var.size(); \
    cudaMalloc(reinterpret_cast<void**>(&ptr_temp), size); \
    CudaCheck(cudaMemcpy(\
        reinterpret_cast<void*>(ptr_temp), \
        h_var_ptr, \
        size, \
        cudaMemcpyHostToDevice); \
            h_var_temp.size() = h_var.size(); \
            h_var_temp.ptr() = reinterpret_cast<type##_GPU*>(ptr_temp)); \
    } \
    delete[] h_var_ptr; \
    }

/**
* @brief 设备端[二维]xtensor
*/
template<typename T>
class xtensor2_GPU {
public:
    T* _data;
    size_t _shape[2];

    __host__ __device__ xtensor2_GPU() = default;

    __host__ __device__ xtensor2_GPU(size_t r, size_t c) {
        _shape[0] = r;
        _shape[1] = c;
    }

    __host__ __device__ T& operator()(size_t row, size_t col) {
        return _data[row * _shape[1] + col];
    }

    __host__ __device__ T& operator[](size_t i) {
        return _data[i];
    }

    __host__ __device__ size_t size() {
        return _shape[1] * _shape[0];
    }

    __host__ __device__ size_t* shape() {
        return _shape;
    }

    __host__ __device__ bool empty() {
        return (size() == 0);
    }
};

/**
 * @brief 用于类内，将数据迁移到自定义设备端【二维】xtensor 的宏。注：xtensor对象变量本身未迁移，需跟随上层类变量一起迁移
 * @param xtensor 主机端的类内【二维】xtensor变量
 * @param xtensor_GPUh 用设备端的类定义的，用于迁移数据的主机端变量
 * @param type xtensor内存储的变量类型
 */
#define Dtrans_xtensor2(xtensor,xtensor_GPUh,type) \
    { \
        auto shape=xtensor.shape(); \
        xtensor_GPUh.shape()[0] = xtensor.shape()[0]; \
        xtensor_GPUh.shape()[1] = xtensor.shape()[1]; \
        type* ptr_temp = new type[xtensor_GPUh.size()]; \
        for (int i = 0; i < xtensor.shape()[0]; ++i) { \
            for (int j = 0; j < xtensor.shape()[1]; ++j) { \
                ptr_temp[i * shape[1] + j] = xtensor(i, j); \
            } \
        } \
        CUdeviceptr ptr; \
        const size_t size_temp = sizeof(type)*xtensor_GPUh.size(); \
        cudaMalloc(reinterpret_cast<void**>(&ptr), size_temp); \
        cudaMemcpy(\
        reinterpret_cast<void*>(ptr), \
        ptr_temp, \
        size_temp, \
        cudaMemcpyHostToDevice); \
        delete[] ptr_temp; \
        xtensor_GPUh._data = reinterpret_cast<type*>(ptr); \
    }
#define TdevXtensor2(h_var,varname,type) Dtrans_xtensor2(h_var.varname,h_var##_temp.varname,type);

 /**
  * @brief 设备端[三维]xtensor
  */
template<typename T>
class xtensor3_GPU {
public:
    T* _data;
    size_t _shape[3];

    __host__ __device__ xtensor3_GPU() = default;

    __host__ __device__ xtensor3_GPU(size_t r, size_t c, size_t h) {
        _shape[0] = r;
        _shape[1] = c;
        _shape[2] = h;
    }

    __host__ __device__ T& operator()(size_t row, size_t col, size_t high) {
        return _data[(row * _shape[1] +col) * _shape[2] + high];
    }

    __host__ __device__ T& operator[](size_t i) {
        return _data[i];
    }

    __host__ __device__ size_t size() {
        return _shape[2] * _shape[1] * _shape[0];
    }

    __host__ __device__ size_t* shape() {
        return _shape;
    }

    __host__ __device__ bool empty() {
        return (size() == 0);
    }
};

/**
 * @brief 用于类内，将数据迁移到自定义设备端【三维】xtensor 的宏。注：xtensor对象变量本身未迁移，需跟随上层类变量一起迁移
 * @param xtensor 主机端的类内【三维】xtensor变量
 * @param xtensor_GPUh 用设备端的类定义的，用于迁移数据的主机端变量
 * @param type xtensor内存储的变量类型
 */
#define Dtrans_xtensor3(xtensor,xtensor_GPUh,type) \
    { \
        auto shape=xtensor.shape(); \
        xtensor_GPUh.shape()[0] = xtensor.shape()[0]; \
        xtensor_GPUh.shape()[1] = xtensor.shape()[1]; \
        xtensor_GPUh.shape()[2] = xtensor.shape()[2]; \
        type* ptr_temp = new type[xtensor_GPUh.size()]; \
        for (int i = 0; i < xtensor.shape()[0]; ++i) { \
            for (int j = 0; j < xtensor.shape()[1]; ++j) { \
                for (int k = 0; k < xtensor.shape()[2]; ++k) { \
                    ptr_temp[(i * shape[1] +j) * shape[2] + k] = xtensor(i, j, k); \
                } \
            } \
        } \
        CUdeviceptr ptr; \
        const size_t size_temp = sizeof(type)*xtensor_GPUh.size(); \
        cudaMalloc(reinterpret_cast<void**>(&ptr), size_temp); \
        cudaMemcpy(\
        reinterpret_cast<void*>(ptr), \
        ptr_temp, \
        size_temp, \
        cudaMemcpyHostToDevice); \
        delete[] ptr_temp; \
        xtensor_GPUh._data = reinterpret_cast<type*>(ptr); \
    }
#define TdevXtensor3(h_var,varname,type) Dtrans_xtensor3(h_var.varname,h_var##_temp.varname,type);

 /**
  * @brief 设备端[四维]xtensor
  */
template<typename T>
class xtensor4_GPU {
public:
    T* _data;
    size_t _shape[4];

    __host__ __device__ xtensor4_GPU() = default;

    __host__ __device__ xtensor4_GPU(size_t s1, size_t s2, size_t s3, size_t s4) {
        _shape[0] = s1;
        _shape[1] = s2;
        _shape[2] = s3;
        _shape[3] = s4;
    }

    __host__ __device__ T& operator()(size_t x1, size_t x2, size_t x3, size_t x4) {
        return _data[((x1 * _shape[1] +x2) * _shape[2] + x3) * _shape[3] + x4];
    }

    __host__ __device__ T& operator[](size_t i) {
        return _data[i];
    }

    __host__ __device__ size_t size() {
        return _shape[3] * _shape[2] * _shape[1] * _shape[0];
    }

    __host__ __device__ size_t* shape() {
        return _shape;
    }

    __host__ __device__ bool empty() {
        return (size() == 0);
    }
};

/**
 * @brief 用于类内，将数据迁移到自定义设备端【四维】xtensor 的宏。注：xtensor对象变量本身未迁移，需跟随上层类变量一起迁移
 * @param xtensor 主机端的类内【四维】xtensor变量
 * @param xtensor_GPUh 用设备端的类定义的，用于迁移数据的主机端变量
 * @param type xtensor内存储的变量类型
 */
#define Dtrans_xtensor4(xtensor,xtensor_GPUh,type) \
    { \
        auto shape = xtensor.shape(); \
        xtensor_GPUh.shape()[0] = shape[0]; \
        xtensor_GPUh.shape()[1] = shape[1]; \
        xtensor_GPUh.shape()[2] = shape[2]; \
        xtensor_GPUh.shape()[3] = shape[3]; \
        type* ptr_temp = new type[xtensor_GPUh.size()]; \
        for (int i = 0; i < xtensor.shape()[0]; ++i) { \
            for (int j = 0; j < xtensor.shape()[1]; ++j) { \
                for (int k = 0; k < xtensor.shape()[2]; ++k) { \
                    for (int l = 0; l < xtensor.shape()[3]; ++l) { \
                        ptr_temp[((i * shape[1] +j) * shape[2] + k) * shape[3] + l] = xtensor(i, j, k, l); \
                    } \
                } \
            } \
        } \
        CUdeviceptr ptr; \
        const size_t size_temp = sizeof(type)*xtensor_GPUh.size(); \
        cudaMalloc(reinterpret_cast<void**>(&ptr), size_temp); \
        cudaMemcpy(\
        reinterpret_cast<void*>(ptr), \
        ptr_temp, \
        size_temp, \
        cudaMemcpyHostToDevice); \
        delete[] ptr_temp; \
        xtensor_GPUh._data = reinterpret_cast<type*>(ptr); \
    }
#define TdevXtensor4(h_var,varname,type) Dtrans_xtensor4(h_var.varname,h_var##_temp.varname,type);

 /**
  * @brief 设备端map
  */
template <typename T1, typename T2>
class dev_map_NeuReact {
public:
    // 访问器：
    __host__ __device__ int hash(T1 key) {
        int key_temp = static_cast<int>(key);
        if (key_temp >= 0) {
            return key_temp;
        }
        else {
            return key_temp + 1574;
        }
    }
    __host__ __device__ T2& operator[](T1 index) {
        return _ptr[hash(index)];
    }
    __host__ __device__ const T2& operator[](T1 index) const {
        return _ptr[hash(index)];
    }
    __host__ __device__ T2& operator[](int index) {
        return _ptr[index];
    }
    __host__ __device__ const T2& operator[](int index) const {
        return _ptr[index];
    }
    __host__ __device__ T2*& ptr() {
        return _ptr;
    }
    __host__ __device__ int& size() {
        return _size;
    }
    __host__ __device__ int& capacity() {
        return _capacity;
    }
    __host__ __device__ bool empty() {
        return (_size == 0);
    }
    __host__ __device__ int count(T1 key) {
        if (_ptr[hash(key)].p_bHasXS == false) {
            return 0;
        }
        else {
            return 1;
        }
    }

private:
    T2* _ptr;
    int _size;
    int _capacity;
}; // dev_map_NeuReact

/**
 * @brief 专用于传递unordered_map <NeutronReactionType, unique_ptr<CDNeutronReaction>>变量的宏
 */
#define Dtrans_dev_map_NeuReact(h_var,h_var_temp) \
	CDNeutronReaction_GPU* h_var_ptr = new CDNeutronReaction_GPU[h_var_temp.capacity()]; \
    for(int i=0;i<h_var_temp.capacity();++i){ \
        h_var_ptr[i].p_bHasXS = false; \
    } \
	for (const auto& value : NeutronReactionType_values) { \
		auto it = h_var.find(value); \
		if (it != h_var.end()) { \
			h_var_ptr[h_var_temp.hash(value)] = h_var.at(value); \
		} \
	} \
	{ \
		CUdeviceptr ptr_temp; \
		const size_t size = sizeof(CDNeutronReaction_GPU) * h_var_temp.capacity(); \
		cudaMalloc(reinterpret_cast<void**>(&ptr_temp), size); \
		CudaCheck(cudaMemcpy( \
			reinterpret_cast<void*>(ptr_temp), \
			h_var_ptr, \
			size, \
			cudaMemcpyHostToDevice)); \
		h_var_temp.size() = h_var.size(); \
		h_var_temp.ptr() = reinterpret_cast<CDNeutronReaction_GPU*>(ptr_temp); \
	} \
	delete[] h_var_ptr;

 /**
  * @brief 布尔常数类
  */
class BoolConstVar {
public:
    bool bIsActiveCyc;
    bool bTallyOn;
    bool bTMS;
    bool bUseDBRC;
    bool bIsMultiGroup;
    bool bIsSabCol;
    bool bIsQuasiStaticD;
    bool bDagmc;
};
#define GPUBool(boolVar) cBoolConstVar_GPU.b##boolVar

// must set the nvcc option "-rdc=true" before using extern in CUDA 
/**
 * @brief 用于定义设备端 [class local vector]的宏组 （已弃用）
 */
#define Ddecl_devVecSize(var) int var##_size;
#define Ddecl_devVec(type, var) \
    type* var;\
    Ddecl_devVecSize(var)

/**
* @brief 用于定义设备端 [global vector]的宏组
*/
#define Dextern_constPtr(d_var) extern CUdeviceptr __constant__ d_var;extern int __constant__ d_var##_size;
#define Ddecl_constPtr(d_var) CUdeviceptr __constant__ d_var;int __constant__ d_var##_size;
#define Dtrans_vector(h_var,d_var) \
    { \
        CUdeviceptr ptr; \
        const size_t size = sizeof(decltype(h_var)::value_type)*h_var.size(); \
        cudaMalloc(reinterpret_cast<void**>(&ptr), size); \
        CudaCheck(cudaMemcpyToSymbol(d_var, &ptr, sizeof(CUdeviceptr))); \
        const size_t size_temp = h_var.size(); \
        CudaCheck(cudaMemcpyToSymbol(d_var##_size, &size_temp, sizeof(int))); \
        cudaMemcpy( \
        reinterpret_cast<void*>(ptr), \
        h_var.data(), \
        size, \
        cudaMemcpyHostToDevice); \
    }

/**
* @brief 用于定义设备端var的宏组
*/
#define Dextern_constVar(d_var,type) extern type __constant__ d_var;
#define Ddecl_constVar(d_var,type) type __constant__ d_var;
#define Dtrans_var(h_var,d_var) CudaCheck(cudaMemcpyToSymbol(d_var, &h_var, sizeof(decltype(h_var))));

/**
* @brief 用于定义设备端var array的宏组
*/
#define Dextern_constVarArray(d_var,type,size) extern type __constant__ d_var[size];
#define Ddecl_constVarArray(d_var,type,size) type __constant__ d_var[size];
#define Dtrans_varArray(h_var,d_var) CudaCheck(cudaMemcpyToSymbol(d_var, h_var, sizeof(decltype(h_var))));

/**
* @brief 用于定义设备端__device__ var的宏组
*/
#define Dextern_devVar(d_var,type) extern type __device__ d_var;
#define Ddecl_devVar(d_var,type) type __device__ d_var;

/**
* @brief 用于将主机端string变量h_var的值传递给设备端，并将指向设备端该变量的指针char*传递给h_var_temp的宏
*/
#define Dtrans_devString(h_var, h_var_temp) \
    { \
        CUdeviceptr temp_string; \
        size_t size_temp_string = sizeof(h_var); \
        cudaMalloc(reinterpret_cast<void**>(&temp_string), size_temp_string); \
        CudaCheck(cudaMemcpy( \
            reinterpret_cast<void*>(temp_string), \
            h_var.c_str(), \
            size_temp_string, \
            cudaMemcpyHostToDevice \
        )); \
        h_var_temp = reinterpret_cast<char*>(temp_string); \
    }

/**
* @brief 用于将主机端CDIndex变量传递给设备端CDIndex_GPU的宏,type必须为CDIndex_GPU
*/
#define Dtrans_CDindex(h_var, d_var, type) \
    { \
        type temp_index; \
        Dtrans_devString(h_var.p_sName, temp_index.p_sName); \
        Dtrans_devVec(h_var.p_vIndex, temp_index.p_vIndex); \
        Dtrans_devVec(h_var.p_vIndexU, temp_index.p_vIndexU); \
        CudaCheck(cudaMemcpyToSymbol(d_var, &temp_index, sizeof(type))); \
    }

 /**
  * @brief 用于将数据由 包含虚函数的基类、派生类 组成的主机端过渡vector传递到GPU
  * @detail 具体用法参见CUDA 样例集devClassPolyMorphism
  * @param h_var 主机端std::vector<virtualClass*>变量
  * @param temp 设备端基类[virtualClass_GPU]指针的指针，即virtualClass_GPU**
  * @param classType 主机端基类类型，默认过渡基类类型为classType##_t，设备端基类类型为classType##_GPU
  */
#define Dtrans_virtualClass(h_var,temp,classType) \
{ \
    classType##_GPU var_temp; \
    var_temp = *(dynamic_cast<classType##_t*>(h_var[i])); \
    CUdeviceptr dev_ptr_temp; \
    const size_t size_temp = sizeof(classType##_GPU); \
    cudaMalloc(reinterpret_cast<void**>(&dev_ptr_temp), size_temp); \
    cudaMemcpy(reinterpret_cast<void*>(dev_ptr_temp), &var_temp, size_temp, cudaMemcpyHostToDevice); \
    SetVirtualFuncPtr << <1, 1 >> > (dev_ptr_temp, var_temp.getType()); \
    cudaDeviceSynchronize(); \
    temp[i] = reinterpret_cast<classType##_GPU*>(dev_ptr_temp); \
}

/**
* @brief 用于将数据由 包含虚函数的基类、派生类 组成的主机端vector到传递到过渡vector
* @detail 具体用法参见CUDA 样例集devClassPolyMorphism
* @param h_var 主机端std::vector<std::unique<virtualClass>>变量
* @param h_var_temp 主机端过渡基类指针向量std::vector<virtualClass*>
* @param classType 主机端基类类型，默认过渡基类类型为classType##_t，设备端基类类型为classType##_GPU
*/
#define Dtrans_virtualClassInterface(h_var,h_var_temp,classType) \
{ \
    classType##_t* var_temp = new classType##_t; \
    *var_temp = *(dynamic_cast<classType*>(h_var[i].get())); \
    h_var_temp.push_back(var_temp); \
}

/**
* @brief 用于为设备端自定义dev_vector分配内存
* @param d_var 设备端dev_vector变量
* @param h_var_temp 主机端临时变量
* @param classType 主机端基类类型
* @param size 向量长度
*/
#define Dmalloc_devVec(d_var,h_var_temp,classType,_size) \
    { \
        CUdeviceptr ptr_temp; \
        size_t size_temp = _size * sizeof(classType); \
        cudaMalloc((void**)&ptr_temp, size_temp); \
        h_var_temp.ptr() = reinterpret_cast<classType*>(ptr_temp); \
        h_var_temp.size() = _size; \
        cudaMemcpyToSymbol(d_var, &h_var_temp, sizeof(dev_vector<classType>)); \
    }

#endif // __GPU_TYPE_H__