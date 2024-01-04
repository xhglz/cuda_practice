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

    //gpu 基于原子操作
    blockSize = 16;
    timer.start_gpu();
    MatmulAtomicAddOnDevice(h_matM, h_matN, d_matP, width);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu with atomicAdd <<<(%d,%d), %d>>>", width, width, width);
    timer.duration_gpu(str);
    compareMat(h_matP, d_matP, size);

    //GPU general implementation <<<64, 16>>>
    timer.start_gpu();
    MatmulSharedOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(with shared memory(static))<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration_gpu(str);
    compareMat(h_matP, d_matP, size);

    //GPU general implementation <<<64, 16>>>
    statMem = false;
    timer.start_gpu();
    MatmulSharedOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(with shared memory(static))<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration_gpu(str);
    compareMat(h_matP, d_matP, size);

    statMem = true;
    timer.start_gpu();
    MatmulSharedConflictOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(with shared memory(static))<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration_gpu(str);
    compareMat(h_matP, d_matP, size);

    //GPU general implementation <<<64, 16>>>
    statMem = false;
    timer.start_gpu();
    MatmulSharedConflictOnDevice(h_matM, h_matN, d_matP, width, blockSize, statMem);
    timer.stop_gpu();
    std::sprintf(str, "matmul in gpu(with shared memory(static))<<<%d, %d>>>", width / blockSize, blockSize);
    timer.duration_gpu(str);
    compareMat(h_matP, d_matP, size);

    return 0;
}
 