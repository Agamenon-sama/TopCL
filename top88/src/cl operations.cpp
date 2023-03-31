#include <iostream>
#include <vector>
#include <cmath>
#include <filesystem>
#include <unordered_map>

#include <assert.h>

#include "Matrix.h"
#include "Env.h"
#include "Kernel.h"
#include "Queue.h"
#include "MemBuffer.h"

namespace fs = std::filesystem;
const fs::path kernelFolder("../res/kernels");

extern std::unordered_map<std::string, clw::MemBuffer*> clBuffers;


// %% PREPARE FINITE ELEMENT ANALYSIS
static float A11[16] = {12,  3, -6, -3,    3, 12,  3,  0,   -6,  3, 12, -3,   -3,  0, -3, 12};
static float A12[16] = {-6, -3,  0,  3,   -3, -6, -3, -6,    0, -3, -6,  3,    3, -6,  3, -6};
static float B11[16] = {-4,  3, -2,  9,    3, -4, -9,  4,   -2, -9, -4, -3,    9,  4, -3, -4};
static float B12[16] = { 2, -3,  4, -9,   -3,  2,  9, -2,    4,  9,  2,  3,   -9, -2,  3,  2};

float* calculateKE(const clw::Env &clenv, clw::Queue &queue) {
    // [A11 A12;A12 A11]
    float *a1 = horizontalConcat4x4Mat(A11, A12);
    float *a2 = horizontalConcat4x4Mat(A12, A11); // A12 should be transposed but it's symmetric so ¯\_(ツ)_/¯ 
    float *a  = verticalConcat8x4Mat(a1, a2);
    delete[] a1; delete[] a2;

    // [B11 B12;B12 B11]
    float *b1 = horizontalConcat4x4Mat(B11, B12);
    float *b2 = horizontalConcat4x4Mat(B12, B11); // same as A12
    float *b  = verticalConcat8x4Mat(b1, b2);
    delete[] b1; delete[] b2;

    // create buffers
    clw::MemBuffer aMatrixBuffer(clenv, clw::MemType::ReadBuffer, sizeof(float) * 8*8, a);
    clw::MemBuffer bMatrixBuffer(clenv, clw::MemType::ReadBuffer, sizeof(float) * 8*8, b);

    // set up kernel
    bool err;
    clw::Kernel kernel(clenv, kernelFolder / "calculate_ke.cl", "calculateKE");
    err = kernel.setKernelArg(0, *clBuffers["nu"]);
    assert(err == true);
    err = kernel.setKernelArg(1, aMatrixBuffer);
    assert(err == true);
    err = kernel.setKernelArg(2, bMatrixBuffer);
    assert(err == true);
    err = kernel.setKernelArg(3, *clBuffers["KE"]);
    assert(err == true);

    size_t workSize = 16;
    err = queue.enqueueNDRK(kernel, &workSize);
    assert(err == true);
    err = queue.enqueueReadCommand(*clBuffers["KE"], sizeof(float) * 8*8, a);
    assert(err == true);

    printMatrix(a, 8, 8);

    delete[] b;
    return a; // a must be deleted by caller
}

Matrix calculateEdofVec(const clw::Env &clenv, clw::Queue &queue, float *nodenrs, const int numOfRows, const int numOfColumns) {
    size_t dataSize = (numOfColumns) * (numOfRows);
    Matrix result;
    result.width = numOfColumns;
    result.height = numOfRows;
    result.data = new float[dataSize];

    // nodenrs(1:end-1,1:end-1)
    for (int i = 0; i < numOfRows; i++) {
        for (int j = 0; j < numOfColumns; j++) {
            result.data[j + i*numOfColumns] = nodenrs[j+i + i*numOfColumns];
        }
    }

    // 2*nodenrs(1:end-1,1:end-1)+1
    clw::MemBuffer inputBuffer(clenv, clw::MemType::ReadBuffer, sizeof(float) * dataSize, result.data);
    // clw::MemBuffer outputBuffer(clenv, clw::MemType::WriteBuffer, sizeof(float) * dataSize);
    clBuffers["edofVec"] = new clw::MemBuffer(clenv, clw::MemType::RWBuffer, sizeof(float) * dataSize);

    clw::Kernel kernel(clenv, kernelFolder / "calculate_edofVec.cl", "calculateEdofVec");
    int err;
    err = kernel.setKernelArg(0, inputBuffer);
    assert(err == true);
    // err = kernel.setKernelArg(1, outputBuffer);
    err = kernel.setKernelArg(1, *clBuffers["edofVec"]);
    assert(err == true);

    size_t workSize = dataSize;
    err = queue.enqueueNDRK(kernel, &workSize);
    assert(err == true);
    err = queue.enqueueReadCommand(*clBuffers["edofVec"], sizeof(float) * dataSize, result.data);
    assert(err == true);

    printMatrix(result);

    reshape(result, dataSize, 1);
    result.width = 1;
    result.height = dataSize;

    return result;
}

Matrix calculateEdofMat(const clw::Env &clenv, clw::Queue &queue, int nely, const Matrix &edofVec) {
    Matrix mat;
    mat.width = 8;
    mat.height = edofVec.width * edofVec.height;
    size_t dataSize = mat.width * mat.height;
    mat.data = new float[dataSize];

    bool err;
    clw::Kernel kernel(clenv, kernelFolder / "calculate_edofMat.cl", "calculateEdofMat");
    clBuffers["edofMat"] = new clw::MemBuffer(clenv, clw::MemType::RWBuffer, sizeof(float) * dataSize);

    err = kernel.setKernelArg(0, *clBuffers["nely"]);
    assert(err == true);
    err = queue.enqueueWriteCommand(*clBuffers["edofVec"], sizeof(float) * mat.height, edofVec.data);
    assert(err == true);
    err = kernel.setKernelArg(1, *clBuffers["edofVec"]);
    assert(err == true);
    err = kernel.setKernelArg(2, *clBuffers["edofMat"]);
    assert(err == true);

    size_t workSize = edofVec.height;
    err = queue.enqueueNDRK(kernel, &workSize);
    assert(err == true);
    err = queue.enqueueReadCommand(*clBuffers["edofMat"], sizeof(float) * dataSize, mat.data);
    assert(err == true);

    return mat;
}

void crazyLoop(const clw::Env &clenv, clw::Queue &queue, Matrix &iH, Matrix &jH, Matrix &sH, size_t nelx, size_t nely, float rmin) {
    // setting which kernel to build and run
    auto filename = kernelFolder;
    std::string kernelName;
    if (std::ceil(rmin) == 1.f) {
        filename /= "crazyLoop_unrolled1.cl";
        kernelName = "crazyLoopUnrolled1";
    }
    else {
        filename /= "crazyLoop.cl";
        kernelName = "crazyLoop";
    }

    bool err;
    clw::Kernel kernel(clenv, filename, kernelName);
    clw::MemBuffer iHBuffer(clenv, clw::MemType::WriteBuffer, sizeof(float) * iH.height*iH.width);
    clw::MemBuffer jHBuffer(clenv, clw::MemType::WriteBuffer, sizeof(float) * jH.height*jH.width);
    clw::MemBuffer sHBuffer(clenv, clw::MemType::WriteBuffer, sizeof(float) * sH.height*sH.width);
    err = kernel.setKernelArg(0, *clBuffers["nelx"]);
    assert(err == true);
    err = kernel.setKernelArg(1, *clBuffers["nely"]);
    assert(err == true);
    err = kernel.setKernelArg(2, *clBuffers["rmin"]);
    assert(err == true);
    err = kernel.setKernelArg(3, iHBuffer);
    assert(err == true);
    err = kernel.setKernelArg(4, jHBuffer);
    assert(err == true);
    err = kernel.setKernelArg(5, sHBuffer);
    assert(err == true);

    size_t workSize[] = {nelx, nely};
    err = queue.enqueueNDRK(kernel, workSize, 2);
    assert(err == true);
    queue.enqueueReadCommand(iHBuffer, sizeof(float) * iH.height*iH.width, iH.data);
    assert(err == true);
    queue.enqueueReadCommand(jHBuffer, sizeof(float) * jH.height*jH.width, jH.data);
    assert(err == true);
    queue.enqueueReadCommand(sHBuffer, sizeof(float) * sH.height*sH.width, sH.data);
    assert(err == true);
}


Matrix calculateSK(const clw::Env &clenv, clw::Queue &queue, size_t nelx, size_t nely, Matrix &xPhys) {
    // todo: make sure not to call these two lines in the loop
    clBuffers["xPhys"] = new clw::MemBuffer(clenv, clw::MemType::RWCopyBuffer, sizeof(float) * xPhys.height * xPhys.width, xPhys.data);
    clBuffers["sK"] = new clw::MemBuffer(clenv, clw::MemType::WriteBuffer, sizeof(float) * nelx * nely);

    Matrix sK;
    // FIXME: this segfaults for some reason
    #if 0
    sK.width = 1;
    sK.height = 64*nelx*nely;
    sK.data = new float[sK.height*sK.width];
    assert(sK.data != nullptr);

    bool err;
    clw::Kernel kernel(clenv, kernelFolder / "calculate_sK.cl", "calculateSK");
    err = kernel.setKernelArg(0, *clBuffers["KE"]);
    assert(err == true);
    err = kernel.setKernelArg(1, *clBuffers["xPhys"]);
    err = queue.enqueueWriteCommand(*clBuffers["xPhys"], sizeof(float) * xPhys.height*xPhys.width, xPhys.data);
    assert(err == true);
    err = kernel.setKernelArg(2, *clBuffers["penal"]);
    assert(err == true);
    err = kernel.setKernelArg(3, *clBuffers["E0"]);
    assert(err == true);
    err = kernel.setKernelArg(4, *clBuffers["Emin"]);
    assert(err == true);
    err = kernel.setKernelArg(5, *clBuffers["sK"]);
    assert(err == true);

    size_t workSize[] = {nelx, nely};
    err = queue.enqueueNDRK(kernel, workSize, 2);
    assert(err == true);
    queue.enqueueReadCommand(*clBuffers["sK"], sizeof(float) * sK.height*sK.width, sK.data);
    assert(err == true);
    #endif

    return sK;
}

void filter1(const clw::Env &clenv, clw::Queue &queue, Matrix &dv) {}

void filter2(const clw::Env &clenv, clw::Queue &queue, Matrix &dv) {
    clBuffers["dv"] = new clw::MemBuffer(clenv, clw::MemType::RWCopyBuffer, sizeof(float) * dv.height*dv.width, dv.data);

    bool err;
    clw::Kernel kernel(clenv, kernelFolder / "filter2.cl", "filter2");
    err = kernel.setKernelArg(0, *clBuffers["H"]);
    assert(err == true);
    err = kernel.setKernelArg(1, *clBuffers["Hs"]);
    assert(err == true);
    err = kernel.setKernelArg(2, *clBuffers["dv"]);
    assert(err == true);

    size_t workSize[] = {dv.width, dv.height};
    err = queue.enqueueNDRK(kernel, workSize, 2);
    assert(err == true);
    err = queue.enqueueReadCommand(*clBuffers["dv"], sizeof(float) * dv.height*dv.width, dv.data);
    assert(err == true);
}

float xPhysSum(const clw::Env &clenv, clw::Queue &queue, Matrix &xPhys) {
    auto width = xPhys.width;
    float sum = 0.f;
    clw::MemBuffer widthBuffer(clenv, clw::MemType::ReadBuffer, sizeof(width), &width);
    clw::MemBuffer partialSumsBuffer(clenv, clw::MemType::WriteBuffer, sizeof(float) * xPhys.height);
    clw::MemBuffer sumBuffer(clenv, clw::MemType::WriteBuffer, sizeof(float));
    float *parts = new float[xPhys.height];

    bool err;
    clw::Kernel kernel(clenv, kernelFolder / "xPhys_sum.cl", "xPhysSum");
    err = kernel.setKernelArg(0, *clBuffers["xPhys"]);
    assert(err == true);
    err = kernel.setKernelArg(1, widthBuffer);
    assert(err == true);
    err = kernel.setKernelArg(2, partialSumsBuffer);
    assert(err == true);
    err = kernel.setKernelArg(3, sumBuffer);
    assert(err == true);

    size_t workSize = xPhys.height;
    err = queue.enqueueNDRK(kernel, &workSize);
    assert(err == true);
    err = queue.enqueueReadCommand(sumBuffer, sizeof(float), &sum);
    assert(err == true);
    err = queue.enqueueReadCommand(partialSumsBuffer, sizeof(float) * xPhys.height, parts);
    assert(err == true);

    for (int i = 0; i < xPhys.height; i++) {
        std::cout << parts[i] << "  ";
    }
    std::cout << "\n";

    delete[] parts;
    return sum;
}