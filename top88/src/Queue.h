#pragma once

#include "Env.h"
#include "Kernel.h"
#include "MemBuffer.h"

namespace clw {
/**
 * @brief Abstraction over cl_command_queue
*/
class Queue {
    public:
    /**
     * @brief Constructor that just create a cl_command_queue object
    */
    Queue(const Env &env);
    ~Queue();

    /**
     * @brief runs a kernel in 1 dimension is global size of 1
     * 
     * It just calls clEnqueueTask
     * 
     * @param kernel the kernel to run
    */
    bool enqueueKernel(const Kernel &kernel) noexcept;
    /**
     * @brief runs a kernel by specifying the work size and dimension
     * 
     * It calls clEnqueueNDRangeKernel. You should check how that function works
     * to understand this one.
     * 
     * @param kernel the kernel to run
     * @param globalWorkSize pointer to number of global execution instances per dimension
     * @param workDim number of dimensions
     * @param globalWorkOffset pointer to offset values per dimensions
     * @param localWorkSize pointer to number of local execution instances per dimension
    */
    bool enqueueNDRK(const Kernel &kernel,
                    const size_t *globalWorkSize,
                    uint32_t workDim = 1,
                    const size_t *globalWorkOffset = nullptr,
                    const size_t *localWorkSize = nullptr) noexcept;
    /**
     * @brief read data from a clbuffer to a memory location
     * 
     * @param outputBuffer clbuffer associated with the kernel output to read from
     * @param dataSize size of data to read in bytes
     * @param writeBuffer pointer to where the data should be written
     * @param offset offset from which to read data in outputBuffer output
     * 
     * @note read operation will block host program until the kernel finishes
     * @note You should use the other overload of this function
    */
    bool enqueueReadCommand(cl_mem outputBuffer, size_t dataSize,
                    void *writeBuffer, size_t offset = 0) noexcept;
    /**
     * @brief read data from a clbuffer to a memory location
     * 
     * @param outputBuffer clbuffer associated with the kernel output to read from
     * @param dataSize size of data to read in bytes
     * @param writeBuffer pointer to where the data should be written
     * @param offset offset from which to read data in outputBuffer output
     * 
     * @note read operation will block host program until the kernel finishes
    */
    bool enqueueReadCommand(MemBuffer &outputBuffer, size_t dataSize,
                    void *writeBuffer, size_t offset = 0) noexcept;
    /**
     * @brief write data to a clbuffer from a memory location
     * 
     * @param inputBuffer clbuffer associated with the kernel that it takes as input
     * @param dataSize size of data to read in bytes
     * @param readBuffer pointer to where the data should be read from
     * @param offset offset from which to write data in inputBuffer
     * 
     * @note write operation is also set to block although not sure if that makes a difference
    */
    bool enqueueWriteCommand(MemBuffer &inputBuffer, size_t dataSize,
                    void *readBuffer, size_t offset = 0) noexcept;
    /**
     * @brief blocks host program until all enqueued tasks finish
     * 
     * Simple wrapper over clFinish
    */
    cl_int finish() noexcept;

    private:
    cl_command_queue _queue;
};
}



