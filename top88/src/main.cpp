#include <iostream>
#include <vector>
#include <cmath>
#include <filesystem>
#include <unordered_map>
#include <future>
#include <exception>

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


#include "cl operations.h"


void close(clw::Queue &queue);
SparseMatrix calculateHs(const SparseMatrix &H);
Matrix calculateIK(const Matrix &edofMat);
Matrix calculateJK(const Matrix &edofMat);
Matrix copyMatrix(const Matrix &mat);

int main(int argc, char *argv[]) {
    clw::Env clenv;
    clw::Queue queue(clenv);

    int nelx = 10, nely = 5, penal = 3, ft = 2;
    float rmin = 0.5f, volfrac = 0.5f;

    if (argc == 7) {
        // Usage: ./top88 nelx nely volfrac penal rmin ft
        try {
            nelx = std::stoi(argv[1]);
            nely = std::stoi(argv[2]);
            volfrac = std::stof(argv[3]);
            penal = std::stoi(argv[4]);
            rmin = std::stof(argv[5]);
            ft = std::stoi(argv[6]);
        }
        catch (std::exception &e) {
            std::cerr << "Unvalid arguments " << e.what() << "\n";
            std::cerr << "Usage:\n\t"
                      << argv[0] << "\n\t"
                      << argv[0] << " nelx nely volfrac penal rmin ft\n";
            return 1;
        }
    }

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

    // iK = reshape(kron(edofMat,ones(8,1))',64*nelx*nely,1);
    auto iKFut = std::async(std::launch::async, calculateIK, edofMat);
    // jK = reshape(kron(edofMat,ones(1,8))',64*nelx*nely,1);
    auto jKFut = std::async(std::launch::async, calculateJK, edofMat);
    

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
    clBuffers["x"] = new clw::MemBuffer(clenv, clw::MemType::RWCopyBuffer, sizeof(float) * x.height*x.width, x.data);
    std::cout << "x =\n";
    printMatrix(x);
    std::cout << "\n";

    // xPhys = x;
    auto xPhys = copyMatrix(x);

    Matrix iK = iKFut.get();
    Matrix jK = jKFut.get();
    std::cout << "iK =\n";
    printMatrix(iK);
    std::cout << "\n";
    std::cout << "jK =\n";
    printMatrix(jK);
    std::cout << "\n";


    auto sK = calculateSK(clenv, queue, nelx, nely, xPhys);
    std::cout << "sK =\n";
    printMatrix(sK);
    std::cout << "\n";
    // auto K = SparseMatrix(iK, jK, sK);

    // uint64_t loop = 0;
    // float change = 1.f;
    // while (change > 0.01f) {
    //     loop++;
        auto ce = calculateCE(clenv, queue, KE, U, edofMat, nelx, nely);
        std::cout << "ce =\n";
        printMatrix(ce);
        std::cout << "\n";
        auto dc = calculateDC(clenv, queue, xPhys, ce);
        std::cout << "dc =\n";
        printMatrix(dc);
        std::cout << "\n";
        // todo: since this is gonna run in a loop check if you neef to free the memory
        // allocated by ones()
        auto dv = ones(nely, nelx); 
        filter(clenv, queue, dv);
        std::cout << "dv =\n";
        printMatrix(dv);
        std::cout << "\n";

        float l1 = 0, l2 = 1e9, move = 0.2;
        clBuffers["move"] = new clw::MemBuffer(clenv, clw::MemType::RWCopyBuffer, sizeof(float), &move);
        // while ((l2-l1)/(l1+l2) > 1e-3) {
            float lmid = 0.5*(l2+l1);
            clBuffers["lmid"] = new clw::MemBuffer(clenv, clw::MemType::RWCopyBuffer, sizeof(float), &move);
            auto xnew = calculateXNew(clenv, queue, dc);
            std::cout << "xnew =\n";
            printMatrix(xnew);
            std::cout << "\n";
            // sum(xPhys(:)) > volfrac*nelx*nely
            float s = xPhysSum(clenv, queue, xPhys);
            std::cout << "s = " << s << "\n";
            if (s > volfrac * nelx * nely) {
                l1 = lmid;
            } else {
                l2 = lmid;
            }
        // }
        float change = calculateChange(clenv, queue, x);
        std::cout << "change = " << change << "\n";
    // }
    



    close(queue);

    return 0;
}

void close(clw::Queue &queue) {
    // start by waiting to finish all enqueued commands cause otherwise, segfault
    auto err = queue.finish(); // should be unnecessary but better be safe than sorry
    assert(err == CL_SUCCESS);

    for (auto &buff : clBuffers) {
        // I originally added this line to make sure I release the buffers
        // before destroying the context in clenv but I guess I don't need it
        // plus it was causing random segfaults
        // Still gonna keep it commented just in case
        // buff.second->~MemBuffer();
        delete buff.second; // not sure if clBuffers.clear() does this, probably not
    }
    clBuffers.clear();
}

Matrix calculateIK(const Matrix &edofMat) {
    Matrix iK;
    iK.width = 1;
    iK.height = 8 * edofMat.height * edofMat.width;
    iK.data = new float[iK.width * iK.height];

    // we just need to repeat each column 8 times
    for (int i = 0, c = 0; i < edofMat.height; i++) {
        for (int j = 0; j < 8; j++, c++) {
            // Manually unrolled the loop
            iK.data[c * 8 + 0] = edofMat.data[i * edofMat.width + 0];
            iK.data[c * 8 + 1] = edofMat.data[i * edofMat.width + 1];
            iK.data[c * 8 + 2] = edofMat.data[i * edofMat.width + 2];
            iK.data[c * 8 + 3] = edofMat.data[i * edofMat.width + 3];
            iK.data[c * 8 + 4] = edofMat.data[i * edofMat.width + 4];
            iK.data[c * 8 + 5] = edofMat.data[i * edofMat.width + 5];
            iK.data[c * 8 + 6] = edofMat.data[i * edofMat.width + 6];
            iK.data[c * 8 + 7] = edofMat.data[i * edofMat.width + 7];
        }
    }

    return iK;
}

Matrix calculateJK(const Matrix &edofMat) {
    Matrix jK;
    jK.width = 1;
    jK.height = 8 * edofMat.height * edofMat.width;
    jK.data = new float[jK.width * jK.height];

    // we just need to repeat each value 8 times
    for(int i = 0; i < edofMat.height * edofMat.width; i++) {
        // Manually unrolled the loop
        jK.data[i * 8 + 0] = edofMat.data[i];
        jK.data[i * 8 + 1] = edofMat.data[i];
        jK.data[i * 8 + 2] = edofMat.data[i];
        jK.data[i * 8 + 3] = edofMat.data[i];
        jK.data[i * 8 + 4] = edofMat.data[i];
        jK.data[i * 8 + 5] = edofMat.data[i];
        jK.data[i * 8 + 6] = edofMat.data[i];
        jK.data[i * 8 + 7] = edofMat.data[i];
    }

    return jK;
}

Matrix copyMatrix(const Matrix &mat) {
    Matrix result;
    result.height = mat.height;
    result.width = mat.width;
    result.data = new float[result.width * result.height];

    for (int i = 0; i < result.width * result.height; i++) {
        result.data[i] = mat.data[i];
    }

    return result;
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



