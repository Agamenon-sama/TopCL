#pragma once

#include <stdint.h>

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
