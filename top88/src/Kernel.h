#pragma once

#include "Env.h"
#include "MemBuffer.h"

#include <string>

namespace clw {
/**
 * @brief Abstraction over cl_kernel and cl_program
*/
class Kernel {
    public:
    /**
     * @brief Contructor for Kernel
     * 
     * This constructor opens the kernel file, compiles it and gets the kernel function
     * specified by @a kernelName. If more than 1 kernel function exists in the file,
     * this constructor won't get it and you will have to create another Kernel object.
     * It is therefore recommended that you write 1 function per file to avoid compiling
     * the file many times.
     * 
     * @param env the Env object that abstracts the cl device and context
     * @param clFile the name of the file that contains the kernel
     * @param kernelName name of kernel function
    */
    Kernel(const Env &env, const std::string &clFile, const std::string &kernelName);
    ~Kernel();

    /**
     * @brief abstraction over clSetKernelArg
     * 
     * @param index index of the kernel parameter. Parameters are index starting from 0.
     * @param argSize size of argument type
     * @param argValue pointer to the data that should be used as argument
     * 
     * @note This function is not used anywhere as the other overload is preferred
    */
    bool setKernelArg(int index, size_t argSize, const void *argValue) noexcept;
    /**
     * @brief abstraction over clSetKernelArg
     * 
     * @param index index of the kernel parameter. Parameters are index starting from 0.
     * @param buffer MemBuffer to be used as argument
    */
    bool setKernelArg(int index, const MemBuffer &buffer) noexcept;

    cl_kernel getKernel() const { return _kernel; } ///< @brief returns underlying cl_kernel

    private:
    cl_kernel _kernel;
    cl_program _program;
};
}


