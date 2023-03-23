#include "Queue.h"

#include <stdexcept>

clw::Queue::Queue(const Env &env) {
    int err;
    _queue = clCreateCommandQueue(env.getContext(), env.getDevice(), 0, &err);
    if (err < 0) {
        throw std::runtime_error("Couldn't make command queue (" + std::to_string(err) + ")");
    }
}

clw::Queue::~Queue() {
    clReleaseCommandQueue(_queue);
}

bool clw::Queue::enqueueKernel(const Kernel &kernel) noexcept {
    if (clEnqueueTask(_queue, kernel.getKernel(), 0, nullptr, nullptr) < 0) {
        return false;
    }
    return true;
}


bool clw::Queue::enqueueNDRK(
    const Kernel &kernel,
    const size_t *globalWorkSize,
    uint32_t workDim,
    const size_t *globalWorkOffset,
    const size_t *localWorkSize) noexcept
{
    int err;
    err = clEnqueueNDRangeKernel(_queue, kernel.getKernel(), workDim,
            globalWorkOffset, globalWorkSize, localWorkSize,
            0, nullptr, nullptr);
    if (err < 0) {
        return false;
    }
    return true;
}

bool clw::Queue::enqueueReadCommand(MemBuffer &outputBuffer, size_t dataSize,
    void *writeBuffer, size_t offset) noexcept
{
    int err = clEnqueueReadBuffer(_queue, outputBuffer.getBuffer(), CL_TRUE, offset, dataSize,
                writeBuffer, 0, nullptr, nullptr);
    if (err < 0) {
        return false;
    }
    return true;
}


bool clw::Queue::enqueueReadCommand(cl_mem outputBuffer, size_t dataSize,
    void *writeBuffer, size_t offset) noexcept 
{
    int err = clEnqueueReadBuffer(_queue, outputBuffer, CL_TRUE, offset, dataSize,
                writeBuffer, 0, nullptr, nullptr);
    if (err < 0) {
        return false;
    }
    return true;
}

bool clw::Queue::enqueueWriteCommand(MemBuffer &inputBuffer, size_t dataSize,
    void *readBuffer, size_t offset) noexcept
{
    int err = clEnqueueWriteBuffer(_queue, inputBuffer.getBuffer(), CL_TRUE, offset, dataSize,
                readBuffer, 0, nullptr, nullptr);
    if (err < 0) {
        return false;
    }
    return true;
}

cl_int clw::Queue::finish() noexcept {
    return clFinish(_queue); // this could fail but I'm ignoring this because it's unlikely
}

