#include <stdio.h>
#include <cuda_runtime.h>

#include "utils.h"
#include "timer.h"
#include "matmul.h"

static char str[100];

int main(void) {  
    Timer timer;
    int width     = 1 << 10;
    int min       = 0;
    int max       = 1;
    int size      = width * width;
    int blockSize = 1;
    bool statMem  = true;

    float *h_matM = (float*)malloc(size * sizeof(float));
    float *h_matN = (float*)malloc(size * sizeof(float));
    float *h_matP = (float*)malloc(size * sizeof(float));
    float *d_matP = (float*)malloc(size * sizeof(float));

    int seed = 1;
    initMatrix(h_matM, size, min, max, seed);
    seed += 1;
    initMatrix(h_matN, size, min, max, seed);
    LOG("Input size is %d x %d", width, width);

    // cpu
    timer.start_cpu();
    MatmulOnHost(h_matM, h_matN, h_matP, width);
    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("matmul in cpu");

    // GPU warmup
    timer.start_gpu();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop_gpu();
    timer.duration_gpu("matmul in gpu(warmup)");

    // GPU general implementation bs = 1
    blockSize = 1;
    timer.start_gpu();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop_gpu();
    timer.duration_gpu("matmul in gpu(bs = 1)");
    // compareMat(h_matP, d_matP, size);

    // GPU general implementation bs = 16
    blockSize = 16;
    timer.start_gpu();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop_gpu();
    timer.duration_gpu("matmul in gpu(bs = 16)");
    // compareMat(h_matP, d_matP, size);

    // GPU general implementation bs = 32
    blockSize = 32;
    timer.start_gpu();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop_gpu();
    std::printf("gpu<<<dimGrid %d, dimBlock %d>>>", width / blockSize, blockSize);
    timer.duration_gpu("matmul in gpu(bs = 32)");
    compareMat(h_matP, d_matP, size);

    // GPU general implementation bs = 64
    blockSize = 64;
    timer.start_gpu();
    MatmulOnDevice(h_matM, h_matN, d_matP, width, blockSize);
    timer.stop_gpu();
    std::printf("gpu<<<dimGrid %d, dimBlock %d>>>", width / blockSize, blockSize);
    timer.duration_gpu("matmul in gpu(bs = 64)");
    compareMat(h_matP, d_matP, size);

    return 0;
}
 