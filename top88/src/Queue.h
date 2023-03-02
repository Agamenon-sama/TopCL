#pragma once

#include "Env.h"
#include "Kernel.h"
#include "MemBuffer.h"

namespace clw {
class Queue {
    public:
    Queue(const Env &env);
    ~Queue();

    bool enqueueKernel(const Kernel &kernel) noexcept;
    bool enqueueNDRK(const Kernel &kernel,
                    const size_t *globalWorkSize, // TODO: check if fixed
                    uint32_t workDim = 1,
                    const size_t *globalWorkOffset = nullptr,
                    const size_t *localWorkSize = nullptr) noexcept;
    bool enqueueReadCommand(cl_mem outputBuffer, size_t dataSize,
                    void *writeBuffer, size_t offset = 0) noexcept;
    bool enqueueReadCommand(MemBuffer outputBuffer, size_t dataSize,
                    void *writeBuffer, size_t offset = 0) noexcept;

    private:
    cl_command_queue _queue;
};
}



