#include <iostream>
#include <vector>
#include <cmath>

#include <assert.h>

#include "Env.h"
#include "Kernel.h"
#include "Queue.h"
#include "MemBuffer.h"


// %% MATERIAL PROPERTIES
float E0 = 1.f;
float Emin = 1e-9f;
float nu = 0.3f;

// %% PREPARE FINITE ELEMENT ANALYSIS
static float A11[16] = {12,  3, -6, -3,    3, 12,  3,  0,   -6,  3, 12, -3,   -3,  0, -3, 12};
static float A12[16] = {-6, -3,  0,  3,   -3, -6, -3, -6,    0, -3, -6,  3,    3, -6,  3, -6};
static float B11[16] = {-4,  3, -2,  9,    3, -4, -9,  4,   -2, -9, -4, -3,    9,  4, -3, -4};
static float B12[16] = { 2, -3,  4, -9,   -3,  2,  9, -2,    4,  9,  2,  3,   -9, -2,  3,  2};


float* horizontalConcat4x4Mat(const float *a, const float *b);
float* verticalConcat8x4Mat(const float *a, const float *b);
float* calculateKE(const clw::Env &clenv, clw::Queue &queue);
std::vector<float> makeVector(float first, float last, uint8_t step = 1);
float* reshape(const std::vector<float> &vec, int numOfRows, int numOfColumns);

int main(int argc, char *argv[]) {
    clw::Env clenv;
    clw::Queue queue(clenv);

    int nelx = 10, nely = 5;

    float *KE = calculateKE(clenv, queue);
    delete[] KE;

    float *nodenrs = reshape(makeVector(1, (1+nelx)*(1+nely)), 1+nely, 1+nelx);
    delete[] nodenrs;

    return 0;
}

float* horizontalConcat4x4Mat(const float *a, const float *b) {
    assert(a != nullptr && b != nullptr);

    float *c = new float[8*4]; // caller must delete this
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            c[i + (j*8)] = a[i + (j*4)];
            c[i+4 + (j*8)] = b[i + (j*4)];
        }
    }

    return c;
}

float* verticalConcat8x4Mat(const float *a, const float *b) {
    assert(a != nullptr && b != nullptr);

    float *c = new float[8*8]; // caller must delete this
    for (int i = 0; i < 8*4; i++) {
        c[i] = a[i];
        c[i + 8*4] = b[i];
    }

    return c;
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
    clw::Kernel kernel(clenv, "../res/kernels/calculate_ke.cl", "calculateKE");
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

    int y = 0;
    for (int i = 0; i < 8*8; i++) {
        std::cout << a[i] << "  ";
        y++;
        if (y == 8) {
            std::cout << "\n";
            y = 0;
        }
    }
    std::cout << "\n";

    delete[] b;
    return a; // a must be deleted by caller
}

std::vector<float> makeVector(float first, float last, uint8_t step) {
    assert(step >= 1);
    if (first >= last) {
        std::swap(first, last);
    }
    size_t size = std::ceil((last - first + 1) / step);
    std::vector<float> vec(size);
    
    for (int i = 0, value = first; i < vec.size(); i++, value += step) {
        vec[i] = value;
    }
    std::cout << "\n";

    return vec;
}

float* reshape(const std::vector<float> &vec, int numOfRows, int numOfColumns) {
    float *result = new float[numOfRows * numOfColumns];
    for (int y = 0; y < numOfRows; y++) {
        for (int x = 0; x < numOfColumns; x++) {
            result[y * numOfColumns + x] =  vec[x * numOfRows + y];
        }
    }

    int y = 0;
    for (int i = 0; i < numOfRows * numOfColumns; i++) {
        std::cout << result[i] << "  ";
        y++;
        if (y == numOfColumns) {
            std::cout << "\n";
            y = 0;
        }
    }
    std::cout << "\n";

    return result; // caller must delete
}

