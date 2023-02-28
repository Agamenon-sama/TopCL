#include <iostream>

#include <assert.h>

#include "Env.h"
#include "Kernel.h"
#include "Queue.h"
#include "MemBuffer.h"


int main(int argc, char *argv[]) {
    clw::Env clenv;
    clw::Kernel kernel(clenv, "../res/kernels/ker.cl", "ker");
    clw::Queue queue(clenv);

    float buff[4] = {1.5f, 0.75f, 1.2f, 2.24f};
    int err;
    // TODO: see if I can abstract buffer creation
    // buffer for input argument
    clw::MemBuffer inBuffer(clenv, clw::MemType::ReadBuffer, sizeof(float) * 4, buff);
    // cl_mem inBuffer = clCreateBuffer(clenv.getContext(),
    //         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 4, buff, &err);
    // if (err < 0) {
    //     perror("Can't create read buffer");
    //     return 1;
    // }
    // buffer for output argument
    clw::MemBuffer outBuffer(clenv, clw::MemType::WriteBuffer, sizeof(float) * 4);
    // cl_mem outBuffer = clCreateBuffer(clenv.getContext(), CL_MEM_WRITE_ONLY,
    //         sizeof(float) * 4, nullptr, &err);
    // if (err < 0) {
    //     perror("Can't create write buffer");
    //     return 1;
    // }
    
    // set those buffers as kernel arguments
    err = kernel.setKernelArg(0, inBuffer);
    assert(err == true);
    err = kernel.setKernelArg(1, outBuffer);
    assert(err == true);
    // kernel.setKernelArg(0, sizeof(cl_mem), &inBuffer);
    // kernel.setKernelArg(1, sizeof(cl_mem), &outBuffer);

    // enqueue the kernel to run
    // queue.enqueueNDRK(kernel);
    err = queue.enqueueKernel(kernel);
    assert(err == true);

    // read result
    float result[4];
    err = queue.enqueueReadCommand(outBuffer, sizeof(float) * 4, result);
    assert(err == true);
    

    // write output
    for (float x : result) {
        std::cout << x << "  ";
    }
    std::cout << "\n";

    return 0;
}
