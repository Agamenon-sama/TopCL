#pragma once

#include <stdint.h>
#include <stddef.h>

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

struct SparseMatrix {
    SparseMatrix() = default;

    SparseMatrix(const std::vector<size_t> &rows, const std::vector<size_t> &columns, const std::vector<float> &values,
            const uint32_t height, const uint32_t width) {
        _construct(rows, columns, values, height, width);
    }

    SparseMatrix(const Matrix &I, const Matrix &J, const Matrix &SV) {
        std::vector<size_t> vI(I.height);
        std::vector<size_t> vJ(J.height);
        std::vector<float> vSV(SV.height);

        for (size_t i = 0; i < I.height; i++) {
            vI[i] = I.data[i];
            vJ[i] = J.data[i];
            vSV[i] = SV.data[i];
        }

        // can't use delegate constructor apparently
        _construct(vI, vJ, vSV, SV.height, SV.height);
    }

    std::vector<float> values;
    std::vector<size_t> columns;
    std::vector<size_t> rowPtrs;

    uint32_t width;
    uint32_t height;

    private:
        void _construct(const std::vector<size_t> &rows, const std::vector<size_t> &columns, const std::vector<float> &values,
                const uint32_t height, const uint32_t width) {
            this->height = height;
            this->width = width;

            rowPtrs.resize(height + 1);

            // temporary matrix that stores all values including 0 to make the algorithm easier
            float *mat = new float[width*height];

            // initialize everything to 0
            for (size_t i = 0; i < width*height; i++) {
                mat[i] = 0;
            }
            // fill the values given as inputs and sum the values with duplicates coordinates
            for (size_t i = 0; i < values.size(); i++) {
                // note the -1 which is to convert from the matlab base 1 indexing to the C++ base 0 indexing
                mat[(rows[i]-1)*width + (columns[i]-1)] += values[i];
            }

            size_t nonZeroID = 0;
            for (size_t r = 0; r < height; r++) {
                rowPtrs[r] = nonZeroID; // storing the index of the first non-zero in the row
                for (size_t c = 0; c < width; c++) {
                    if (mat[r*width + c] != 0.f) {
                        this->values.emplace_back(mat[r*width + c]);
                        this->columns.emplace_back(c);
                        nonZeroID++;
                    }
                }
            }
            rowPtrs[height] = nonZeroID;

            delete[] mat;
        }
};

struct CooSparseMatrix {
    std::vector<float> values;
    std::vector<size_t> rows;
    std::vector<size_t> columns;

    CooSparseMatrix(const Matrix &iMat, const Matrix &jMat, const Matrix &kMat) {
        for (int i = 0; i < iMat.height*iMat.width; i++) {
            rows.emplace_back(iMat.data[i]);
        }
        for (int i = 0; i < jMat.height*jMat.width; i++) {
            columns.emplace_back(jMat.data[i]);
        }
        for (int i = 0; i < kMat.height*kMat.width; i++) {
            values.emplace_back(kMat.data[i]);
        }
    }
};


void printMatrix(const Matrix &mat);
void printMatrix(const float *vec, const int width, const int height);
void printSparse(const SparseMatrix &spmat);
float* horizontalConcat4x4Mat(const float *a, const float *b);
float* verticalConcat8x4Mat(const float *a, const float *b);
std::vector<float> makeVector(float first, float last, uint8_t step = 1);
float* reshape(const std::vector<float> &vec, int numOfRows, int numOfColumns);
void reshape(Matrix &mat, int numOfRows, int numOfColumns);
Matrix zeros(const int height, const int width);
Matrix ones(const int height, const int width);
void setdiff(std::vector<float> &a, const std::vector<float> &b);
Matrix repmat(float value, int numberOfRows, int numberOfColumns);
