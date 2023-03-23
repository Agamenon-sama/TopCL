#include "Kernel.h"

#include <fstream>
#include <iostream>

clw::Kernel::Kernel(const Env &env, const std::string &clFile, const std::string &kernelName) {
    // open the .cl file
    std::fstream programFile(clFile, std::ios::in);
    if (!programFile.is_open()) {
        throw std::runtime_error("Can't open file " + clFile);
    }

    // read content
    programFile.seekg(0, std::ios::end);
    size_t fileSize = programFile.tellg();
    programFile.seekg(0);
    char *fileBuffer = new char[fileSize + 1];
    fileBuffer[fileSize] = '\0';
    programFile.read(fileBuffer, fileSize);
    programFile.close();

    // create program
    int err;
    _program = clCreateProgramWithSource(env.getContext(), 1,
        (const char**) &fileBuffer, &fileSize, &err);
    if (err < 0) {
        throw std::runtime_error("Couldn't create a program from " + clFile
                                +  " (" + std::to_string(err) + ")");
    }
    delete [] fileBuffer; // don't need this anymore

    // build program
    err = clBuildProgram(_program, 0, nullptr, nullptr, nullptr, nullptr);
    if (err < 0) {
        // get size of log info
        size_t logSize;
        clGetProgramBuildInfo(_program, env.getDevice(), CL_PROGRAM_BUILD_LOG,
                0, nullptr, &logSize);
        char *logBuffer = new char[logSize + 1];
        logBuffer[logSize] = '\0';
        clGetProgramBuildInfo(_program, env.getDevice(), CL_PROGRAM_BUILD_LOG,
                logSize + 1, logBuffer, nullptr);
        std::cerr << "Failed to build program\nlog: " << logBuffer << "\n";
        delete [] logBuffer;
        throw std::runtime_error("Failed to build program (" + std::to_string(err) + ")");
    }

    // create kernel
    _kernel = clCreateKernel(_program, kernelName.c_str(), &err);
    if (err < 0) {
        throw std::runtime_error("Couldn't create a kernel for " + clFile
                                + " (" + std::to_string(err) + ")");
    }
}

clw::Kernel::~Kernel() {
    clReleaseKernel(_kernel);
    clReleaseProgram(_program);
}

bool clw::Kernel::setKernelArg(int index, const MemBuffer &buffer) noexcept {
    // if (clSetKernelArg(_kernel, index, sizeof(cl_mem), buffer.getBuffer()) < 0) {
    cl_mem x = buffer.getBuffer();
    // int err = clSetKernelArg(_kernel, index, sizeof(cl_mem), &buffer._buffer);
    int err = clSetKernelArg(_kernel, index, sizeof(cl_mem), &x);
    if (err  < 0) {
        std::cerr << errorStr(err) << "\n";
        return false;
    }
    return true;
}

bool clw::Kernel::setKernelArg(int index, size_t argSize, const void *argValue) noexcept {
    auto err = clSetKernelArg(_kernel, index, argSize, argValue);
    if (err < 0) {
        return false;
    }
    return true;
}


