#ifndef __GPU_ERRORCTRL_H__
#define  __GPU_ERRORCTRL_H__

#include <vector>
#include <string>
#include<unordered_map>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include"GPUPrint.h"
#include"GPUBaseFunc.h"

// 定义枚举元素，自动映射
#define ENUM_GPU_ERROR_TYPES \
    ENUM_VALUE(success) \
    ENUM_VALUE(vecOverRange) \
    ENUM_VALUE(divByZero)

// 声明枚举类。不可修改，自动映射
#define ENUM_VALUE(name) name,
enum class eGPUErrorType { ENUM_GPU_ERROR_TYPES };
#undef ENUM_VALUE

// 自动生成映射表的宏
#define GENERATE_ENUM_MAP(ENUM_LIST) const std::unordered_map<eGPUErrorType, std::string> errorToString = { ENUM_LIST };

extern const std::unordered_map<eGPUErrorType, std::string> errorToString;

//**************************************************
//	class def
//**************************************************
// 设备端错误日志类
class CDGPULog {
private:
    char* _log;        // 存储错误日志的缓冲区
    size_t _capacity;     // 缓冲区总容量
    size_t _used;        // 已使用的字节数（设备端指针）
    bool _isFull;

public:
    // 构造函数
    CDGPULog() = default;
    CDGPULog(size_t capacity);

    // 初始化日志（清空缓冲区）
    void initialize(size_t capacity = 1024 * 1024 * 32);

    __host__ __device__ __inline__ void reset() { _used = 0; _isFull = false; }

    __host__ __device__ __inline__ bool isFull() { return _isFull; }

    // 设备端记录错误信息
    __device__ size_t record(const char* error_str);

    // 从设备端复制错误数据到主机端
    void copyFromDev(std::vector<char>& h_ErrorLog);

    // 打印错误日志到文件
    void printToFile(const char* filename = "error.log");

    std::string getLog(size_t startPos, char* log);
};

class CDGPUErrorCtrl {
private:
    // 设备端错误记录数组
    int* _errorCodes;              // 错误代码数组
    size_t* _errorParticleIds;        // 错误粒子编号数组
    size_t* _errorLogStartPos;      // 错误对应的log在日志中的开始位置
    size_t _errorCapacity; // 错误容量
    size_t _errorCount;    // 错误计数器
    size_t _errorCyc;       // 第一次发生错误的代数
    CDGPULog _errorLog;

public:
    // 主机端构造函数
    CDGPUErrorCtrl() = default;
    CDGPUErrorCtrl(size_t ErrorCapacity, size_t logCapacity);

    void initialize(size_t ErrorCapacity = 102400, size_t logCapacity= 1024 * 1024 * 16);

    // 禁止拷贝
    CDGPUErrorCtrl(const CDGPUErrorCtrl&) = delete;
    CDGPUErrorCtrl& operator=(const CDGPUErrorCtrl&) = delete;

    // 获取设备端指针(用于传递给核函数)
    __host__ __device__ __inline__ int* getErrorCodePtr() { return _errorCodes; }
    __host__ __device__ __inline__ size_t* getErrorParticleIdPtr() { return _errorParticleIds; }
    __host__ __device__ __inline__ size_t* getErrorLogStartPos() { return _errorLogStartPos; }
    __host__ __device__ __inline__ size_t getErrorCount() { return _errorCount; }
    __host__ __device__ __inline__ size_t getErrorCapacity() { return _errorCapacity; }

    // 判断是否有错误发生
    __host__ __device__ __inline__ bool isErrorOccur() { return (_errorCount != 0); }

    // 从设备端复制错误数据到主机端
    void copyFromDev(std::vector<int>& h_ErrorCode, std::vector<size_t>& h_ErrorThread, std::vector<size_t>& h_ErrorLogStartPos);

    // 重置设备端错误记录
    void reset();

    // 打印错误信息
    void print(const std::string& filename = "GPUErrors.log");

    // 设备端错误记录函数
    __device__ void record(eGPUErrorType errorCode, size_t particleId, const char* log);
};

__device__ void strMergeDevice(char* output, int outputSize,
    const char* cstr, const char* filename, int line);

//**************************************************
//	extern var
//**************************************************

extern __device__ CDGPUErrorCtrl OGPUErrorCtrl;

#define RECORD_ERROR(errorCode, particleId, cstr) \
{ \
    char buffer[256]; \
    strMergeDevice(buffer, sizeof(buffer), cstr, __FILE_GPU__, __LINE__); \
    OGPUErrorCtrl.record(errorCode, particleId, buffer); \
}


#endif //__GPU_ERRORCTRL_H__