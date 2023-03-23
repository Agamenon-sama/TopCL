#pragma once

#include "Env.h"

namespace clw {
enum class MemType {
    ReadBuffer = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    WriteBuffer = CL_MEM_WRITE_ONLY,
    RWBuffer = CL_MEM_READ_WRITE
};

class MemBuffer {
    public:
    MemBuffer() = default;
    MemBuffer(const Env &env, MemType flag, size_t size, void *data = nullptr);
    ~MemBuffer();

    cl_mem getBuffer() const { return _buffer; }

    private:
    cl_mem _buffer;
};
}
