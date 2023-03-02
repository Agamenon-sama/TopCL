#include <iostream>

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

int main(int argc, char *argv[]) {
    clw::Env clenv;
    // clw::Kernel kernel(clenv, "../res/kernels/ker.cl", "ker");
    clw::Queue queue(clenv);

    float *KE = calculateKE(clenv, queue);
    delete[] KE;

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
    float coefficient = (1/(1-(nu*nu)))/24;

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

    // nu*[B11 B12;B12 B11]
    clw::MemBuffer nuBuffer(clenv, clw::MemType::ReadBuffer, sizeof(float), &nu);
    clw::MemBuffer bMatrixBuffer(clenv, clw::MemType::ReadBuffer, sizeof(float) * 8*8, b);
    clw::MemBuffer bResMatrixBuffer(clenv, clw::MemType::WriteBuffer, sizeof(float) * 8*8);
    bool err;
    clw::Kernel multKernel(clenv, "../res/kernels/scalar_matrix_mult.cl", "scalarMatrixMult");
    err = multKernel.setKernelArg(0, nuBuffer);
    assert(err == true);
    err = multKernel.setKernelArg(1, bMatrixBuffer);
    assert(err == true);
    err = multKernel.setKernelArg(2, bResMatrixBuffer);
    assert(err == true);

    size_t workUnits = 16;
    err = queue.enqueueNDRK(multKernel, &workUnits);
    assert(err == true);
    err = queue.enqueueReadCommand(bResMatrixBuffer, sizeof(float) * 8*8, b);
    assert(err == true);

    y = 0;
    for (int i = 0; i < 8*8; i++) {
        std::cout << b[i] << "  ";
        y++;
        if (y == 8) {
            std::cout << "\n";
            y = 0;
        }
    }
    std::cout << "\n";

    // [A11 A12;A12' A11]+nu*[B11 B12;B12 B11]
    clw::MemBuffer aMatrixBuffer(clenv, clw::MemType::ReadBuffer, sizeof(float) * 8*8, a);
    clw::MemBuffer bMatrixBuffer2(clenv, clw::MemType::ReadBuffer, sizeof(float) * 8*8, b);
    clw::Kernel addKernel(clenv, "../res/kernels/matrix_matrix_add.cl", "matrixMatrixAdd");
    err = addKernel.setKernelArg(0, aMatrixBuffer);
    assert(err == true);
    err = addKernel.setKernelArg(1, bMatrixBuffer2);
    assert(err == true);
    err = addKernel.setKernelArg(2, bResMatrixBuffer);
    assert(err == true);

    err = queue.enqueueNDRK(addKernel, &workUnits);
    assert(err == true);
    err = queue.enqueueReadCommand(bResMatrixBuffer, sizeof(float) * 8*8, b);
    assert(err == true);

    y = 0;
    for (int i = 0; i < 8*8; i++) {
        std::cout << b[i] << "  ";
        y++;
        if (y == 8) {
            std::cout << "\n";
            y = 0;
        }
    }
    std::cout << "\n";

    // 1/(1-nu^2)/24*([A11 A12;A12' A11]+nu*[B11 B12;B12 B11])
    clw::MemBuffer coefficientBuffer(clenv, clw::MemType::ReadBuffer, sizeof(float), &coefficient);
    clw::MemBuffer matrixBuffer(clenv, clw::MemType::ReadBuffer, sizeof(float) * 8*8, b);
    clw::MemBuffer finalResultBuffer(clenv, clw::MemType::WriteBuffer, sizeof(float) * 8*8);
    err = multKernel.setKernelArg(0, coefficientBuffer);
    assert(err == true);
    err = multKernel.setKernelArg(1, matrixBuffer);
    assert(err == true);
    err = multKernel.setKernelArg(2, finalResultBuffer);
    assert(err == true);

    err = queue.enqueueNDRK(multKernel, &workUnits);
    assert(err == true);
    err = queue.enqueueReadCommand(finalResultBuffer, sizeof(float) * 8*8, b);
    assert(err == true);

    delete[] a;

    std::cout << "KE =\n";
    y = 0;
    for (int i = 0; i < 8*8; i++) {
        std::cout << b[i] << "  ";
        y++;
        if (y == 8) {
            std::cout << "\n";
            y = 0;
        }
    }
    std::cout << "\n";

    return b; // caller must delete b
}
