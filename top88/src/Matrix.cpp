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

    printMatrix(result, numOfColumns, numOfRows);

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

    printMatrix(mat);
    std::cout << "\n";
}


Matrix zeros(const int height, const int width) {
    Matrix mat;
    mat.data = new float[width * height];
    mat.width = width;
    mat.height = height;

    for (int i = 0; i < width * height; i++) {
        mat.data[i] = 0;
    }

    return mat;
}

Matrix ones(const int height, const int width) {
    Matrix mat;
    mat.data = new float[width * height];
    mat.width = width;
    mat.height = height;

    for (int i = 0; i < width * height; i++) {
        mat.data[i] = 1;
    }

    return mat;
}

void printMatrix(const Matrix &mat) {
    int y = 0;
    for (int i = 0; i < mat.height * mat.width; i++) {
        std::cout << mat.data[i] << "  ";
        y++;
        if (y == mat.width) {
            std::cout << "\n";
            y = 0;
        }
    }
    std::cout << "\n";
}
void printMatrix(const float *vec, const int width, const int height) {
    int y = 0;
    for (int i = 0; i < height * width; i++) {
        std::cout << vec[i] << "  ";
        y++;
        if (y == width) {
            std::cout << "\n";
            y = 0;
        }
    }
    std::cout << "\n";
}

void setdiff(std::vector<float> &a, const std::vector<float> &b) {
    for (size_t i = 0; i < b.size(); i++) {
        for (auto it = a.begin(); it != a.end(); it++) {
            if (*it == b[i]) {
                a.erase(it);
                it--;
            }
        }
    }
}

void printSparse(const SparseMatrix &spmat) {
    std::cout << "Compressed Row Sparse (rows = " << spmat.height << ", cols = " << spmat.width
              << ", nnz = " << spmat.values.size() << " [" << ((float)spmat.values.size() / (spmat.height*spmat.width)) * 100
              << "%])\n\n";

    for (size_t i = 0; i < spmat.height; i++) {
        size_t nonZeroStart = spmat.rowPtrs[i];
        size_t nonZeroEnd = spmat.rowPtrs[i+1];
        for (size_t nonZeroID = nonZeroStart; nonZeroID < nonZeroEnd; nonZeroID++) {
            std::cout << "  (" << i << ", " << spmat.columns[nonZeroID] << ") -> " << spmat.values[nonZeroID] << "\n";
        }
    }

    std::cout << "\n";
}

Matrix repmat(float value, int numberOfRows, int numberOfColumns) {
    Matrix mat;
    mat.height = numberOfRows;
    mat.width = numberOfColumns;
    mat.data = new float[numberOfColumns * numberOfRows];

    for (size_t i = 0; i < numberOfColumns * numberOfRows; i++) {
        mat.data[i] = value;
    }

    return mat;
}

