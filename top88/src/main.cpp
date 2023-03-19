#include <iostream>
#include <vector>
#include <cmath>
#include <filesystem>

#include <assert.h>

#include "Matrix.h"
#include "Env.h"
#include "Kernel.h"
#include "Queue.h"
#include "MemBuffer.h"

namespace fs = std::filesystem;
const fs::path kernelFolder("../res/kernels");

// %% MATERIAL PROPERTIES
float E0 = 1.f;
float Emin = 1e-9f;
float nu = 0.3f;

// %% PREPARE FINITE ELEMENT ANALYSIS
static float A11[16] = {12,  3, -6, -3,    3, 12,  3,  0,   -6,  3, 12, -3,   -3,  0, -3, 12};
static float A12[16] = {-6, -3,  0,  3,   -3, -6, -3, -6,    0, -3, -6,  3,    3, -6,  3, -6};
static float B11[16] = {-4,  3, -2,  9,    3, -4, -9,  4,   -2, -9, -4, -3,    9,  4, -3, -4};
static float B12[16] = { 2, -3,  4, -9,   -3,  2,  9, -2,    4,  9,  2,  3,   -9, -2,  3,  2};


float* calculateKE(const clw::Env &clenv, clw::Queue &queue);
Matrix calculateEdofVec(const clw::Env &clenv, clw::Queue &queue, float *nodenrs, const int numOfRows, const int numOfColumns);
Matrix calculateEdofMat(const clw::Env &clenv, clw::Queue &queue, int nely, const Matrix &edofVec);
void crazyLoop(const clw::Env &clenv, clw::Queue &queue, Matrix &iH, Matrix &jH, Matrix &sH, size_t nelx, size_t nely, float rmin);

int main(int argc, char *argv[]) {
    clw::Env clenv;
    clw::Queue queue(clenv);

    int nelx = 10, nely = 5;
    float rmin = 0.5f;

    std::cout << "KE =\n";
    float *KE = calculateKE(clenv, queue);
    delete[] KE;

    float *nodenrs = reshape(makeVector(1, (1+nelx)*(1+nely)), 1+nely, 1+nelx);
    Matrix edofVec = calculateEdofVec(clenv, queue, nodenrs, nely, nelx);
    Matrix edofMat = calculateEdofMat(clenv, queue, nely, edofVec);
    delete[] nodenrs;

    // F = sparse(2,1,-1,2*(nely+1)*(nelx+1),1);
    SparseMatrix F({2}, {1}, {-1}, 2*(nely+1)*(nelx+1), 1);
    std::cout << "F =\n";
    printSparse(F);

    // U = zeros(2*(nely+1)*(nelx+1),1);
    Matrix U = zeros(2*(nely+1)*(nelx+1), 1);
    std::cout << "U =\n";
    printMatrix(U);

    // given the nature of the input, the following union can be replaced with a simple insertion
    // fixeddofs = union([1:2:2*(nely+1)],[2*(nelx+1)*(nely+1)]);
    std::vector<float> fixeddofs = makeVector(1, 2 * (nely+1), 2);
    fixeddofs.emplace_back(2 * (nelx+1) * (nely+1));
    std::cout << "fixeddofs =\n";
    printMatrix(fixeddofs.data(), fixeddofs.size(), 1);

    // alldofs = [1:2*(nely+1)*(nelx+1)]
    std::vector<float> freedofs = makeVector(1, 2 * (nely+1) * (nelx+1));
    std::cout << "alldofs =\n";
    printMatrix(freedofs.data(), freedofs.size(), 1);

    // freedofs = setdiff(alldofs,fixeddofs);
    // to avoid copying alldofs, I'm sending it to be modified
    setdiff(freedofs, fixeddofs);
    std::cout << "freedofs =\n";
    printMatrix(freedofs.data(), freedofs.size(), 1);

    // %% PREPARE FILTER

    // iH = ones(nelx*nely*(2*(ceil(rmin)-1)+1)^2,1);
    Matrix iH = ones(nelx * nely * std::pow((2 * (std::ceil(rmin) - 1) + 1), 2), 1);
    // jH = ones(size(iH));
    Matrix jH = ones(iH.height, iH.width);
    // sH = zeros(size(iH));
    Matrix sH = zeros(iH.height, iH.width);

    crazyLoop(clenv, queue, iH, jH, sH, nelx, nely, rmin);
    std::cout << "iH =\n";
    printMatrix(iH);
    std::cout << "\n";
    std::cout << "jH =\n";
    printMatrix(jH);
    std::cout << "\n";
    std::cout << "sH =\n";
    printMatrix(sH);
    std::cout << "\n";


    return 0;
}


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
    clw::MemBuffer nuBuffer(clenv, clw::MemType::ReadBuffer, sizeof(float), &nu);
    clw::MemBuffer aMatrixBuffer(clenv, clw::MemType::ReadBuffer, sizeof(float) * 8*8, a);
    clw::MemBuffer bMatrixBuffer(clenv, clw::MemType::ReadBuffer, sizeof(float) * 8*8, b);
    clw::MemBuffer resultMatrixBuffer(clenv, clw::MemType::WriteBuffer, sizeof(float) * 8*8);

    // set up kernel
    bool err;
    clw::Kernel kernel(clenv, kernelFolder / "calculate_ke.cl", "calculateKE");
    err = kernel.setKernelArg(0, nuBuffer);
    assert(err == true);
    err = kernel.setKernelArg(1, aMatrixBuffer);
    assert(err == true);
    err = kernel.setKernelArg(2, bMatrixBuffer);
    assert(err == true);
    err = kernel.setKernelArg(3, resultMatrixBuffer);
    assert(err == true);

    size_t workSize = 16;
    err = queue.enqueueNDRK(kernel, &workSize);
    assert(err == true);
    err = queue.enqueueReadCommand(resultMatrixBuffer, sizeof(float) * 8*8, a);
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
    clw::MemBuffer outputBuffer(clenv, clw::MemType::WriteBuffer, sizeof(float) * dataSize);
    clw::Kernel kernel(clenv, kernelFolder / "calculate_edofVec.cl", "calculateEdofVec");
    int err;
    err = kernel.setKernelArg(0, inputBuffer);
    assert(err == true);
    err = kernel.setKernelArg(1, outputBuffer);
    assert(err == true);

    size_t workSize = dataSize;
    err = queue.enqueueNDRK(kernel, &workSize);
    assert(err == true);
    err = queue.enqueueReadCommand(outputBuffer, sizeof(float) * dataSize, result.data);
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
    clw::MemBuffer nelyBuffer(clenv, clw::MemType::ReadBuffer, sizeof(int), &nely);
    clw::MemBuffer edofVecBuffer(clenv, clw::MemType::ReadBuffer, sizeof(float) * mat.height, edofVec.data);
    clw::MemBuffer outputBuffer(clenv, clw::MemType::WriteBuffer, sizeof(float) * dataSize);
    err = kernel.setKernelArg(0, nelyBuffer);
    assert(err == true);
    err = kernel.setKernelArg(1, edofVecBuffer);
    assert(err == true);
    err = kernel.setKernelArg(2, outputBuffer);
    assert(err == true);

    size_t workSize = edofVec.height;
    err = queue.enqueueNDRK(kernel, &workSize);
    assert(err == true);
    err = queue.enqueueReadCommand(outputBuffer, sizeof(float) * dataSize, mat.data);
    assert(err == true);

    printMatrix(mat);

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
    clw::MemBuffer nelxBuffer(clenv, clw::MemType::ReadBuffer, sizeof(int), &nelx);
    clw::MemBuffer nelyBuffer(clenv, clw::MemType::ReadBuffer, sizeof(int), &nely);
    clw::MemBuffer rminBuffer(clenv, clw::MemType::ReadBuffer, sizeof(float), &rmin);
    clw::MemBuffer iHBuffer(clenv, clw::MemType::WriteBuffer, sizeof(float) * iH.height*iH.width);
    clw::MemBuffer jHBuffer(clenv, clw::MemType::WriteBuffer, sizeof(float) * jH.height*jH.width);
    clw::MemBuffer sHBuffer(clenv, clw::MemType::WriteBuffer, sizeof(float) * sH.height*sH.width);
    err = kernel.setKernelArg(0, nelxBuffer);
    assert(err == true);
    err = kernel.setKernelArg(1, nelyBuffer);
    assert(err == true);
    err = kernel.setKernelArg(2, rminBuffer);
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
