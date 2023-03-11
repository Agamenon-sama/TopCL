#pragma once

#include <stdint.h>

#include <vector>

struct Matrix {
    Matrix() {
        data = nullptr;
        width = 0;
        height = 0;
    }
    ~Matrix() {
        if (!data) {
            delete[] data;
            data = nullptr;
        }
    }
    
    float *data;
    uint32_t width;
    uint32_t height;
};

float* horizontalConcat4x4Mat(const float *a, const float *b);
float* verticalConcat8x4Mat(const float *a, const float *b);
std::vector<float> makeVector(float first, float last, uint8_t step = 1);
float* reshape(const std::vector<float> &vec, int numOfRows, int numOfColumns);
void reshape(Matrix &mat, int numOfRows, int numOfColumns);

