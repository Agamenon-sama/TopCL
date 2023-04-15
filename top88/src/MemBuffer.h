#pragma once

#include "Env.h"

namespace clw {
/**
 * @brief Abstraction over cl_mem_flags
 * 
 * MemType are the combinations of cl_mem_flags values that are clw rapper consider
 * 
 * @note In the following, @a input means input data for the kernel 
 * and @a output means output for the kernel
*/
enum class MemType {
    /**
     * @brief @b input buffer to send data with.
     * It will copy data from @a data pointer in the constructor
     * 
     * Use this if you only want to send data to the kernel to work with
     * without the kernel changing it
    */
    ReadBuffer = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    /**
     * @brief @b output buffer to read the output of the kernel from
     * 
     * Use this if you only want to read the output
    */
    WriteBuffer = CL_MEM_WRITE_ONLY,
    /**
     * @brief can be used for both reading and writing from and to the kernel
     * 
     * Use this if you want to read and write to the buffer but you don't want to initilize it
     * with a value. Basically use it if the buffer is going to be first used as a write buffer
     * but then that result will be used as a read buffer.
    */
    RWBuffer = CL_MEM_READ_WRITE,
    /**
     * @brief can be used for both reading and writing from and to the kernel
     * but requires and non null value for @a data pointer in the constructor
     * 
     * Use this if you want to read and write to the buffer but you want to initilize it
     * with a value first. Basically use it if the kernel is going to read its data and then
     * write the output to the same buffer.
    */
    RWCopyBuffer = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR
};

/**
 * @brief Abstraction over cl_mem
*/
class MemBuffer {
    public:
    MemBuffer() = default;
    /**
     * @brief MemBuffer constructor
     * 
     * @param env the Env object that abstracts the cl device and context
     * @param flag type of buffer to create
     * @param size size of the buffer in bytes
     * @param data in case of read buffer, data points to some data that will be copied to the buffer before sending
     * 
     * @sa MemType
    */
    MemBuffer(const Env &env, MemType flag, size_t size, void *data = nullptr);
    ~MemBuffer();

    cl_mem getBuffer() const { return _buffer; } ///< @brief returns the cl_mem underneath

    private:
    cl_mem _buffer;
};
}
