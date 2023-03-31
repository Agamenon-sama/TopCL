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

std::unordered_map<std::string, clw::MemBuffer*> clBuffers;

// %% MATERIAL PROPERTIES
float E0 = 1.f;
float Emin = 1e-9f;
float nu = 0.3f;

float* calculateKE(const clw::Env &clenv, clw::Queue &queue);
Matrix calculateEdofVec(const clw::Env &clenv, clw::Queue &queue, float *nodenrs, const int numOfRows, const int numOfColumns);
Matrix calculateEdofMat(const clw::Env &clenv, clw::Queue &queue, int nely, const Matrix &edofVec);
void crazyLoop(const clw::Env &clenv, clw::Queue &queue, Matrix &iH, Matrix &jH, Matrix &sH, size_t nelx, size_t nely, float rmin);
Matrix calculateSK(const clw::Env &clenv, clw::Queue &queue, size_t nelx, size_t nely, Matrix &xPhys);
void filter1(const clw::Env &clenv, clw::Queue &queue, Matrix &dv);
void filter2(const clw::Env &clenv, clw::Queue &queue, Matrix &dv);
float xPhysSum(const clw::Env &clenv, clw::Queue &queue, Matrix &xPhys);

void close(clw::Queue &queue);
SparseMatrix calculateHs(const SparseMatrix &H);

int main(int argc, char *argv[]) {
    clw::Env clenv;
    clw::Queue queue(clenv);

    int nelx = 10, nely = 5, penal = 3, ft = 2;
    float rmin = 0.5f, volfrac = 0.5f;

    auto filter = ft == 1 ? filter1 : filter2;

    clBuffers["nu"] = new clw::MemBuffer(clenv, clw::MemType::ReadBuffer, sizeof(float), &nu);
    clBuffers["nelx"] = new clw::MemBuffer(clenv, clw::MemType::ReadBuffer, sizeof(int), &nelx);
    clBuffers["nely"] = new clw::MemBuffer(clenv, clw::MemType::ReadBuffer, sizeof(int), &nely);
    clBuffers["penal"] = new clw::MemBuffer(clenv, clw::MemType::ReadBuffer, sizeof(int), &penal);
    clBuffers["rmin"] = new clw::MemBuffer(clenv, clw::MemType::ReadBuffer, sizeof(float), &rmin);
    clBuffers["E0"] = new clw::MemBuffer(clenv, clw::MemType::ReadBuffer, sizeof(float), &E0);
    clBuffers["Emin"] = new clw::MemBuffer(clenv, clw::MemType::ReadBuffer, sizeof(float), &Emin);
    clBuffers["KE"] = new clw::MemBuffer(clenv, clw::MemType::WriteBuffer, sizeof(float) * 8*8);

    std::cout << "KE =\n";
    float *KE = calculateKE(clenv, queue);
    delete[] KE;

    float *nodenrs = reshape(makeVector(1, (1+nelx)*(1+nely)), 1+nely, 1+nelx);
    Matrix edofVec = calculateEdofVec(clenv, queue, nodenrs, nely, nelx);
    Matrix edofMat = calculateEdofMat(clenv, queue, nely, edofVec);
    std::cout << "edofMat =\n";
    printMatrix(edofMat);
    std::cout << "\n";
    delete[] nodenrs;

    // F = sparse(2,1,-1,2*(nely+1)*(nelx+1),1);
    SparseMatrix F({2}, {1}, {-1}, 2*(nely+1)*(nelx+1), 1);
    std::cout << "F =\n";
    printSparse(F);

    // U = zeros(2*(nely+1)*(nelx+1),1);
    Matrix U = zeros(2*(nely+1)*(nelx+1), 1);

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

    // H = sparse(iH,jH,sH);
    SparseMatrix H(iH, jH, sH);
    auto Hs = calculateHs(H);
    std::cout << "Hs =\n";
    printSparse(Hs);
    clBuffers["H"] = new clw::MemBuffer(clenv, clw::MemType::ReadBuffer, sizeof(float) * H.values.size(), H.values.data());
    clBuffers["Hs"] = new clw::MemBuffer(clenv, clw::MemType::ReadBuffer, sizeof(float) * Hs.values.size(), Hs.values.data());

    // x = repmat(volfrac,nely,nelx);
    auto x = repmat(volfrac, nely, nelx);
    std::cout << "x =\n";
    printMatrix(x);
    std::cout << "\n";

    // xPhys = x;
    auto xPhys = x;


    auto sK = calculateSK(clenv, queue, nelx, nely, xPhys);
    // std::cout << "sK =\n";
    // printMatrix(x);
    // std::cout << "\n";
    // auto K = SparseMatrix(iK, jK, sK);

    // uint64_t loop = 0;
    // float change = 1.f;
    // while (change > 0.01f) {
    //     loop++;
        // todo: since this is gonna run in a loop check if you neef to free the memory
        // allocated by ones()
        auto dv = ones(nely, nelx); 
        filter(clenv, queue, dv);
        std::cout << "dv =\n";
        printMatrix(dv);
        std::cout << "\n";

        float l1 = 0, l2 = 1e9, move = 0.2;
        // while ((l2-l1)/(l1+l2) > 1e-3) {
            float lmid = 0.5*(l2+l1);
            // sum(xPhys(:)) > volfrac*nelx*nely
            float s = xPhysSum(clenv, queue, xPhys);
            std::cout << "s = " << s << "\n";
            if (s > volfrac * nelx * nely) {
                l1 = lmid;
            } else {
                l2 = lmid;
            }
        // }
    // }
    



    close(queue);

    return 0;
}

void close(clw::Queue &queue) {
    // start by waiting to finish all enqueued commands cause otherwise, segfault
    auto err = queue.finish(); // should be unnecessary but better be safe than sorry
    assert(err == CL_SUCCESS);

    for (auto &buff : clBuffers) {
        buff.second->~MemBuffer();
        delete buff.second; // not sure if clBuffers.clear() does this, probably not
    }
    clBuffers.clear();
}


SparseMatrix calculateHs(const SparseMatrix &H) {
    // I'm profiting from the fact that H is always a diagonal matrix
    // so the sum of the rows is just the only non 0 value
    SparseMatrix Hs;
    Hs.width = H.width;
    Hs.height = H.height;
    Hs.values.resize(H.values.size());
    Hs.columns.resize(H.columns.size());
    Hs.rowPtrs.resize(H.rowPtrs.size());

    for (size_t i = 0; i < H.columns.size(); i++) {
        Hs.values[i] = H.values[i];
        Hs.columns[i] = 0;
        Hs.rowPtrs[i] = i;
    }
    Hs.rowPtrs[Hs.rowPtrs.size()-1] = Hs.rowPtrs[Hs.rowPtrs.size() - 2] + 1;

    return Hs;
}



