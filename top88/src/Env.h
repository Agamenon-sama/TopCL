#pragma once

#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <string>

/**
 * @brief Wrapper for OpenCL
 * 
 * @author Chamseddine Hachani (Agamenon) <chams.eddine.hachanii@gmail.com>
*/
namespace clw {
/**
 * @brief Abstraction over cl_platform_id, cl_device_id and cl_context
*/
class Env {
    public:
    /**
     * @brief Gets the first platform and gpu device returned by the appropriate cl functions
     * and create a context
    */
    Env();
    ~Env();

    cl_device_id getDevice() const { return _device; }
    cl_context getContext() const { return _context; }

    private:
    cl_platform_id _platform;
    cl_device_id _device;
    cl_context _context;
};

/**
 * @brief Creates a human readable version of the error return codes of all OpenCL errors
 * 
 * @param error an error code returned by an OpenCL function
 * 
 * @return a string version of the enum name in source code and the error number
*/
std::string errorStr(cl_int error);
}
