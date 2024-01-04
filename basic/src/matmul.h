#ifndef _MATMUL_H_
#define _MATMUL_H_

void initMatrix(float *data, int size, int low, int hight, int seed);

void initMatrixSigned(float *data, int size, float low, float high, int seed);

void printMat(float *data, int size);

void compareMat(float *h_data, float *d_data, int size);

void MatmulOnHost(float *M, float *N, float *P, int width);

void MatmulOnDevice(float *M_host, float *N_host, float *P_host, int width, int blockSize);

void MatmulSharedOnDevice(float *M_host, float *N_host, float *P_host, int width, int blockSize, bool staticMem);

void MatmulAtomicAddOnDevice(float *M_host, float *N_host, float *P_host, int width);

void MatmulSharedConflictOnDevice(float *M_host, float *N_host, float *P_host, int width, int blockSize, bool staticMem);
#endif