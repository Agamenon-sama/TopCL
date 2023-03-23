#pragma once

#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <string>

namespace clw {
class Env {
    public:
    Env();
    ~Env();

    cl_device_id getDevice() const { return _device; }
    cl_context getContext() const { return _context; }

    private:
    cl_platform_id _platform;
    cl_device_id _device;
    cl_context _context;
};

std::string errorStr(cl_int error);
}
