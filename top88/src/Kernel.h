#pragma once

#include "Env.h"
#include "MemBuffer.h"

#include <string>

namespace clw {
class Kernel {
    public:
    Kernel(const Env &env, const std::string &clFile, const std::string &kernelName);
    ~Kernel();

    bool setKernelArg(int index, size_t argSize, const void *argValue) noexcept;
    bool setKernelArg(int index, const MemBuffer &buffer) noexcept;

    cl_kernel getKernel() const { return _kernel; }

    private:
    cl_kernel _kernel;
    cl_program _program;
};
}


