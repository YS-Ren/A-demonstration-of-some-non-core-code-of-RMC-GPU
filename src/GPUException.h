#ifndef __GPU_EXCEPTION_H__
#define  __GPU_EXCEPTION_H__

#include <stdexcept>
#include <iostream>
#include <string>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <optix.h>
#include<optix_stubs.h>

enum class eHostErrorType {

};

class Exception : public std::runtime_error
{
public:
    Exception(const char* msg)
        : std::runtime_error(msg)
    {
    }

    Exception(OptixResult res, const char* msg)
        : std::runtime_error(createMessage(res, msg).c_str())
    {
    }

private:
    std::string createMessage(OptixResult res, const char* msg)
    {
        std::ostringstream out;
        out << optixGetErrorName(res) << ": " << msg;
        return out.str();
    }
};

inline void cudaCheck(cudaError_t error, const char* call, const char* file, unsigned int line)
{
    if (error != cudaSuccess)
    {
        std::stringstream ss;
        ss << "CUDA call (" << call << " ) failed with error: '"
            << cudaGetErrorString(error) << "' (" << file << ":" << line << ")\n";
        throw Exception(ss.str().c_str());
    }
}
#define CUDA_CHECK( call ) ::cudaCheck( call, #call, __FILE__, __LINE__ )

inline void cudaSyncCheck(const char* file, unsigned int line)
{
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::stringstream ss;
        ss << "CUDA error on synchronize with error '"
            << cudaGetErrorString(error) << "' (" << file << ":" << line << ")\n";
        throw Exception(ss.str().c_str());
    }
}
#define CUDA_SYNC_CHECK() ::cudaSyncCheck( __FILE__, __LINE__ )

inline void optixCheck(OptixResult res, const char* call, const char* file, unsigned int line)
{
    if (res != OPTIX_SUCCESS)
    {
        std::stringstream ss;
        ss << "Optix call '" << call << "' failed: " << file << ':' << line << ")\n";
        throw Exception(res, ss.str().c_str());
    }
}
#define OPTIX_CHECK( call )                                                    \
    ::optixCheck( call, #call, __FILE__, __LINE__ )



#endif //__GPU_EXCEPTION_H__