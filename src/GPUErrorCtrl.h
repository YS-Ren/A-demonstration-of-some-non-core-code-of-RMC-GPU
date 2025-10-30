#ifndef __GPU_ERRORCTRL_H__
#define  __GPU_ERRORCTRL_H__

#include <vector>
#include <string>
#include<unordered_map>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include"GPUPrint.h"
#include"GPUBaseFunc.h"

// ����ö��Ԫ�أ��Զ�ӳ��
#define ENUM_GPU_ERROR_TYPES \
    ENUM_VALUE(success) \
    ENUM_VALUE(vecOverRange) \
    ENUM_VALUE(divByZero)

// ����ö���ࡣ�����޸ģ��Զ�ӳ��
#define ENUM_VALUE(name) name,
enum class eGPUErrorType { ENUM_GPU_ERROR_TYPES };
#undef ENUM_VALUE

// �Զ�����ӳ���ĺ�
#define GENERATE_ENUM_MAP(ENUM_LIST) const std::unordered_map<eGPUErrorType, std::string> errorToString = { ENUM_LIST };

extern const std::unordered_map<eGPUErrorType, std::string> errorToString;

//**************************************************
//	class def
//**************************************************
// �豸�˴�����־��
class CDGPULog {
private:
    char* _log;        // �洢������־�Ļ�����
    size_t _capacity;     // ������������
    size_t _used;        // ��ʹ�õ��ֽ������豸��ָ�룩
    bool _isFull;

public:
    // ���캯��
    CDGPULog() = default;
    CDGPULog(size_t capacity);

    // ��ʼ����־����ջ�������
    void initialize(size_t capacity = 1024 * 1024 * 32);

    __host__ __device__ __inline__ void reset() { _used = 0; _isFull = false; }

    __host__ __device__ __inline__ bool isFull() { return _isFull; }

    // �豸�˼�¼������Ϣ
    __device__ size_t record(const char* error_str);

    // ���豸�˸��ƴ������ݵ�������
    void copyFromDev(std::vector<char>& h_ErrorLog);

    // ��ӡ������־���ļ�
    void printToFile(const char* filename = "error.log");

    std::string getLog(size_t startPos, char* log);
};

class CDGPUErrorCtrl {
private:
    // �豸�˴����¼����
    int* _errorCodes;              // �����������
    size_t* _errorParticleIds;        // �������ӱ������
    size_t* _errorLogStartPos;      // �����Ӧ��log����־�еĿ�ʼλ��
    size_t _errorCapacity; // ��������
    size_t _errorCount;    // ���������
    size_t _errorCyc;       // ��һ�η�������Ĵ���
    CDGPULog _errorLog;

public:
    // �����˹��캯��
    CDGPUErrorCtrl() = default;
    CDGPUErrorCtrl(size_t ErrorCapacity, size_t logCapacity);

    void initialize(size_t ErrorCapacity = 102400, size_t logCapacity= 1024 * 1024 * 16);

    // ��ֹ����
    CDGPUErrorCtrl(const CDGPUErrorCtrl&) = delete;
    CDGPUErrorCtrl& operator=(const CDGPUErrorCtrl&) = delete;

    // ��ȡ�豸��ָ��(���ڴ��ݸ��˺���)
    __host__ __device__ __inline__ int* getErrorCodePtr() { return _errorCodes; }
    __host__ __device__ __inline__ size_t* getErrorParticleIdPtr() { return _errorParticleIds; }
    __host__ __device__ __inline__ size_t* getErrorLogStartPos() { return _errorLogStartPos; }
    __host__ __device__ __inline__ size_t getErrorCount() { return _errorCount; }
    __host__ __device__ __inline__ size_t getErrorCapacity() { return _errorCapacity; }

    // �ж��Ƿ��д�����
    __host__ __device__ __inline__ bool isErrorOccur() { return (_errorCount != 0); }

    // ���豸�˸��ƴ������ݵ�������
    void copyFromDev(std::vector<int>& h_ErrorCode, std::vector<size_t>& h_ErrorThread, std::vector<size_t>& h_ErrorLogStartPos);

    // �����豸�˴����¼
    void reset();

    // ��ӡ������Ϣ
    void print(const std::string& filename = "GPUErrors.log");

    // �豸�˴����¼����
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