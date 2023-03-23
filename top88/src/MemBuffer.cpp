#include "MemBuffer.h"

#include <stdexcept>

clw::MemBuffer::MemBuffer(const Env &env, MemType flag, size_t size, void *data) {
    int err;
    if (flag == MemType::ReadBuffer || flag == MemType::RWBuffer) {
        _buffer = clCreateBuffer(env.getContext(), (cl_mem_flags)flag, size, data, &err);
    }
    else if (flag == MemType::WriteBuffer) {
        _buffer = clCreateBuffer(env.getContext(), (cl_mem_flags)flag, size, nullptr, &err);
    }
    else {
        throw std::runtime_error("Invalid MemType used to create OpenCL buffer");
    }

    if (err < 0) {
        throw std::runtime_error("Can't create OpenCL buffer: " + errorStr(err));
    }
}

clw::MemBuffer::~MemBuffer() {
    clReleaseMemObject(_buffer);
}
