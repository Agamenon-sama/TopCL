#include "Matrix.h"

#include <assert.h>

#include <iostream>
#include <vector>
#include <cmath>

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
    assert(numOfColumns*numOfRows == vec.size());
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

void reshape(Matrix &mat, int numOfRows, int numOfColumns) {
    assert(numOfColumns*numOfRows == mat.width*mat.height);
    float *result = new float[numOfRows * numOfColumns];
    for (int c = 0, k = 0; c < mat.width; c++) {
        for (int r = 0; r < mat.height; r++, k++) {
            result[k] = mat.data[c + r*mat.width];
        }
    }
    delete[] mat.data;
    mat.data = result;

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
}

