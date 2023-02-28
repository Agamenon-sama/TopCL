#include "Env.h"

#include <stdexcept>

clw::Env::Env() {
    int err;
    
    err = clGetPlatformIDs(1, &_platform, nullptr);
    if (err < 0) {
        throw std::runtime_error("Couldn't find an OpenCL platform (" + std::to_string(err) + ")");
    }

    err = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_GPU, 1, &_device, NULL);
    if (err < 0) {
       throw std::runtime_error("Couldn't access any device (" + std::to_string(err) + ")");
    }

    _context = clCreateContext(nullptr, 1, &_device, nullptr, nullptr, &err);
    if (err < 0) {
       throw std::runtime_error("Couldn't create a context (" + std::to_string(err) + ")");
    }
}

clw::Env::~Env() {
    clReleaseContext(_context);
    clReleaseDevice(_device);
}
