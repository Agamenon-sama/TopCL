#pragma once

#include "Matrix.h"
#include "Env.h"
#include "Kernel.h"
#include "Queue.h"
#include "MemBuffer.h"

float* calculateKE(const clw::Env &clenv, clw::Queue &queue);
Matrix calculateEdofVec(const clw::Env &clenv, clw::Queue &queue, float *nodenrs, const int numOfRows, const int numOfColumns);
Matrix calculateEdofMat(const clw::Env &clenv, clw::Queue &queue, int nely, const Matrix &edofVec);
void crazyLoop(const clw::Env &clenv, clw::Queue &queue, Matrix &iH, Matrix &jH, Matrix &sH, size_t nelx, size_t nely, float rmin);
Matrix calculateSK(const clw::Env &clenv, clw::Queue &queue, size_t nelx, size_t nely, Matrix &xPhys);
void filter1(const clw::Env &clenv, clw::Queue &queue, Matrix &dv);
void filter2(const clw::Env &clenv, clw::Queue &queue, Matrix &dv);
float xPhysSum(const clw::Env &clenv, clw::Queue &queue, Matrix &xPhys);
Matrix calculateCE(const clw::Env &clenv, clw::Queue &queue, float *KE, Matrix &U, const Matrix &edofMat, int nelx, int nely);
Matrix calculateDC(const clw::Env &clenv, clw::Queue &queue, Matrix &xPhys, Matrix &ce);
float calculateC(const clw::Env &clenv, clw::Queue &queue, Matrix &xPhys, Matrix &ce);
Matrix calculateXNew(const clw::Env &clenv, clw::Queue &queue, const Matrix &dc);
float calculateChange(const clw::Env &clenv, clw::Queue &queue, const Matrix &x);
