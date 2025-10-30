#include "GPUErrorCtrl.h"

#include <iostream>
#include <fstream>

#include "GPUPrint.h"

//**************************************************
//	class def
//**************************************************
// class CDGPULog
CDGPULog::CDGPULog(size_t capacity) {
    initialize(capacity);
}

// ��ʼ����־����ջ�������
void CDGPULog::initialize(size_t capacity) {
    _capacity = capacity;
    _used = 0;
    _isFull = false;
    cudaMalloc(reinterpret_cast<void**>(&_log), capacity);
}

// �豸�˼�¼������Ϣ
__device__ size_t CDGPULog::record(const char* error_str) {
    // �����ַ������ȣ���������ֹ����
    size_t len = 0;
    while (error_str[len] != '\0') len++;

    // ԭ���Ե�������ʹ���ֽ���
    size_t start_idx = atomicAdd(&_used, len + 1);  // +1 ���ڻ��з�

    // ����Ƿ����㹻�ռ�
    if (start_idx + len + 1 >= _capacity) {
        _isFull = true;
        return 0;  // �ռ䲻�㣬����¼
    }

    // �����ַ�������־������
    for (size_t i = 0; i < len; i++) {
        _log[start_idx + i] = error_str[i];
    }

    // ��ӻ��з�
    _log[start_idx + len] = '\n';
    return start_idx;
}

// ���豸�˸��ƴ������ݵ�������
void CDGPULog::copyFromDev(std::vector<char>& h_ErrorLog) {
    // ������ʱ������
    h_ErrorLog.resize(_capacity);

    // ���豸�˸�������
    cudaMemcpy(h_ErrorLog.data(), _log, _capacity * sizeof(char), cudaMemcpyDeviceToHost);
}

// ��ӡ������־���ļ�
//void CDGPULog::printToFile(const char* filename) {
//
//    if (_used == 0) {
//        printf("No errors logged.\n");
//        return;
//    }
//
//    // ���������˻������������豸����
//    char* host_log = new char[_used + 1];
//    cudaMemcpy(host_log, _log, _used, cudaMemcpyDeviceToHost);
//    host_log[_used] = '\0';
//
//    // д���ļ�
//    FILE* file = fopen(filename, "w");
//    if (file) {
//        fwrite(host_log, 1, _used, file);
//        fclose(file);
//        printf("Error log written to %s\n", filename);
//    }
//    else {
//        printf("Failed to open file %s for writing\n", filename);
//    }
//    delete[] host_log;
//}

std::string CDGPULog::getLog(size_t startPos, char* log) {
    if (log == nullptr || startPos >= strlen(log)) {
        return "";
    }

    char* startPtr = log + startPos;
    char* newlinePtr = strchr(startPtr, '\n');

    return (newlinePtr != nullptr)
        ? std::string(startPtr, newlinePtr)
        : std::string(startPtr);
}


//class CDGPUErrorCtrl
// ��ʼ������
CDGPUErrorCtrl::CDGPUErrorCtrl(size_t ErrorCapacity, size_t logCapacity) {
    initialize(ErrorCapacity, logCapacity);
}

void CDGPUErrorCtrl::initialize(size_t ErrorCapacity, size_t logCapacity) {
    _errorCapacity = ErrorCapacity;
    _errorCount = 0;
    _errorCyc = 0;
    // �����豸�ڴ�
    cudaMalloc(&_errorCodes, _errorCapacity * sizeof(int));
    cudaMalloc(&_errorParticleIds, _errorCapacity * sizeof(size_t));
    cudaMalloc(&_errorLogStartPos, _errorCapacity * sizeof(size_t));

    // ��ʼ���豸�ڴ�
    cudaMemset(_errorCodes, 0, _errorCapacity * sizeof(int));
    cudaMemset(_errorParticleIds, 0, _errorCapacity * sizeof(size_t));
    cudaMemset(_errorLogStartPos, 0, _errorCapacity * sizeof(size_t));

    _errorLog.initialize(logCapacity);
}

// ���豸�˸��ƴ������ݵ�������
void CDGPUErrorCtrl::copyFromDev(std::vector<int>& h_ErrorCode, std::vector<size_t>& h_ErrorThread, std::vector<size_t>& h_ErrorLogStartPos) {
    // ���Ʋ�������������
    _errorCount = (_errorCount > _errorCapacity) ? _errorCapacity : _errorCount;

    if (_errorCount > 0) {
        // ������ʱ������
        h_ErrorCode.resize(_errorCount);
        h_ErrorThread.resize(_errorCount);
        h_ErrorLogStartPos.resize(_errorCount);

        // ���豸�˸�������
        cudaMemcpy(h_ErrorCode.data(), _errorCodes, _errorCount * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ErrorThread.data(), _errorParticleIds, _errorCount * sizeof(size_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ErrorLogStartPos.data(), _errorLogStartPos, _errorCount * sizeof(size_t), cudaMemcpyDeviceToHost);
    }
}

// �����豸�˴����¼
void CDGPUErrorCtrl::reset() {
    _errorCount = 0;
    _errorCyc = 0;
    cudaMemset(_errorCodes, 0, _errorCapacity * sizeof(int));
    cudaMemset(_errorParticleIds, 0, _errorCapacity * sizeof(size_t));
    cudaMemset(_errorLogStartPos, 0, _errorCapacity * sizeof(size_t));
    _errorLog.reset();
}

// ��ӡ������Ϣ
void CDGPUErrorCtrl::print(const std::string& filename) {
    if (!isErrorOccur()) {
        std::stringstream ss;
        ss << "No errors recorded.";
        OLogger.info(ss.str(), "CDErrorCtrl");
        return;
    }

    // �����˴����¼����(���ڴ��豸�˸�������)
    std::vector<int> h_ErrorCode;
    std::vector<size_t> h_ErrorThread;
    std::vector<size_t> h_ErrorLogStartPos;
    copyFromDev(h_ErrorCode, h_ErrorThread, h_ErrorLogStartPos);

    std::vector<char> h_ErrorLog;
    _errorLog.copyFromDev(h_ErrorLog);

    // write to file
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::stringstream ss;
        ss << "Failed to open error log file: " << filename;
        OLogger.error(ss.str());
        return;
    }

    for (size_t i = 0; i < h_ErrorCode.size(); ++i) {
        outFile << "Error["<<i<<"] [ParticleId] " << h_ErrorThread[i] << " \t[Type] ";
        // ʹ��ӳ�����Ҷ�Ӧ���ַ���
        eGPUErrorType error = static_cast<eGPUErrorType>(h_ErrorCode[i]);
        auto it = errorToString.find(error);
        outFile << it->second.c_str();
        std::string log = _errorLog.getLog(h_ErrorLogStartPos[i], h_ErrorLog.data());
        outFile << " [Position] " << log.c_str() << std::endl;
    }

    outFile.close();
    {
        std::stringstream ss;
        ss << "Error log written to: " << filename;
        OLogger.info(ss.str(), "CDErrorCtrl");
    }
    
    // print summary
    {
        std::stringstream ss;
        ss << "Error Summary (" << h_ErrorCode.size() << " errors):";
        OLogger.header(ss.str(), "CDErrorCtrl");
    }
    for (size_t i = 0; i < std::min(h_ErrorCode.size(), static_cast<size_t>(5)); ++i) {
        std::stringstream ss;
        ss << "Error[" << i << "], [ParticleId] " << h_ErrorThread[i] << " [Type] ";
        // ʹ��ӳ�����Ҷ�Ӧ���ַ���
        eGPUErrorType error = static_cast<eGPUErrorType>(h_ErrorCode[i]);
        auto it = errorToString.find(error);
        ss << it->second.c_str();
        OLogger.main(ss.str());
    }

    if (h_ErrorCode.size() > 5) {
        std::stringstream ss;
        ss << "  ... and " << (h_ErrorCode.size() - 5) << " more errors.";
        OLogger.main(ss.str());
    }
}

// �豸�˴����¼����
__device__ void CDGPUErrorCtrl::record(eGPUErrorType errorCode, size_t particleId, const char* log) {

    size_t startPos = _errorLog.record(log);
    
    // ʹ��ԭ�Ӳ�����ȡ�����¼λ��
    size_t index = atomicAdd(&_errorCount, 1);
    _errorLogStartPos[index] = startPos;

    if ((index < _errorCapacity) && (!_errorLog.isFull())) {
        // ��¼���������߳�ID
        _errorCodes[index] = static_cast<int>(errorCode);
        _errorParticleIds[index] = particleId;
    }
}

__device__ void strMergeDevice(char* output, int outputSize,
    const char* cstr, const char* filename, int line) {
    // �ֶ�ʵ�ָ�ʽ��������ʹ�ñ�׼��
    int pos = 0;

    // �����ļ���
    const char* f = filename;
    while (*f && pos < outputSize - 1) {
        output[pos++] = *f++;
    }

    // ���ð��
    if (pos < outputSize - 1) output[pos++] = ':';

    // ����кţ��ֶ�ת���������ַ�����
    int num = line;
    char numStr[16];
    int numLen = 0;

    // �����к�Ϊ0�����
    if (num == 0) {
        numStr[numLen++] = '0';
    }
    else {
        // ת���������ַ���
        int temp = num;
        while (temp > 0) {
            numLen++;
            temp /= 10;
        }
        temp = num;
        for (int i = numLen - 1; i >= 0; i--) {
            numStr[i] = '0' + (temp % 10);
            temp /= 10;
        }
    }

    // ����к��ַ���
    for (int i = 0; i < numLen && pos < outputSize - 1; i++) {
        output[pos++] = numStr[i];
    }

    // ��ӵ�Ϳո�
    if (pos < outputSize - 1) output[pos++] = ' ';
    if (pos < outputSize - 1) output[pos++] = '[';
    if (pos < outputSize - 1) output[pos++] = 'D';
    if (pos < outputSize - 1) output[pos++] = 'e';
    if (pos < outputSize - 1) output[pos++] = 't';
    if (pos < outputSize - 1) output[pos++] = 'a';
    if (pos < outputSize - 1) output[pos++] = 'i';
    if (pos < outputSize - 1) output[pos++] = 'l';
    if (pos < outputSize - 1) output[pos++] = ']';
    if (pos < outputSize - 1) output[pos++] = ' ';

    // ���ԭʼ�ַ���
    const char* s = cstr;
    while (*s && pos < outputSize - 1) {
        output[pos++] = *s++;
    }

    // ����ַ���������
    output[pos] = '\0';
}

//**************************************************
//	extern var
//**************************************************

__device__ CDGPUErrorCtrl OGPUErrorCtrl;

// ʹ�ú�����ӳ����Զ�ӳ��
#define ENUM_VALUE(name) {eGPUErrorType::name, #name},
GENERATE_ENUM_MAP(ENUM_GPU_ERROR_TYPES)
#undef ENUM_VALUE