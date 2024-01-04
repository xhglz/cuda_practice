#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_stream.cuh"
#include "gelu.cuh"
#include "utils.h"
#include "timer.h"
#include "matmul.h"


void sleep_test() {
    Timer timer;

    int width  = 1 << 10; // 4,096
    int size   = width * width;
    float low  = -1.0;
    float high = 1.0;
    int seed   = 0;

    int blockSize = 16;
    int taskCnt = 5;
    bool statMem = true;
    char str[100];

    /* 初始化 */
    float *src_host;
    float *tar_host;
    cudaMallocHost(&src_host, size * sizeof(float));
    cudaMallocHost(&tar_host, size * sizeof(float));

    seed += 1;
    initMatrixSigned(src_host, size, low, high, seed);
    LOG("Input size is %d", size);


    /* GPU warmup */
    timer.start_gpu();
    SleepSingleStream(src_host, tar_host, width, blockSize, taskCnt);
    timer.stop_gpu();


    /* 1 stream，处理一次memcpy，以及n个kernel */
    blockSize = 16;
    timer.start_gpu();
    SleepSingleStream(src_host, tar_host, width, blockSize, taskCnt);
    timer.stop_gpu();
    std::sprintf(str, "sleep <<<(%2d,%2d), (%2d,%2d)>>>, %2d stream, %2d memcpy, %2d kernel",
                 width / blockSize, width / blockSize, blockSize, blockSize,
                 1, 1, taskCnt);
    timer.duration_gpu(str);

    /* n stream，处理一次memcpy，以及n个kernel */
    timer.start_gpu();
    SleepMultiStream(src_host, tar_host, width, blockSize, taskCnt);
    timer.stop_gpu();
    std::sprintf(str, "sleep <<<(%2d,%2d), (%2d,%2d)>>>, %2d stream, %2d memcpy, %2d kernel",
                 width / blockSize, width / blockSize, blockSize, blockSize,
                 taskCnt, 1, taskCnt);
    timer.duration_gpu(str);
}
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // 需要先确认自己的GPU是否支持overlap计算
    if (!prop.deviceOverlap) {
        LOG("device does not support overlap");
    } else {
        LOG("device supports overlap");
    }

    sleep_test();
    // 这里供大家自由发挥。建议花一些在这里做调度的练习。根据ppt里面的方案实际编写几个测试函数。举几个例子在这里
    // e.g. 一个stream处理: H2D, 多个kernel，D2H。之后多个stream进行overlap
    // e.g. 一个stream处理: H2D, 大kernel，小kernel, D2H。之后多个stream进行overlap
    // e.g. 一个stream处理: H2D, 大kernel, H2D, 小kernel, D2H。之后多个stream进行overlap
    // e.g. 一个stream处理: H2D, 小kernel, H2D, 大kernel, D2H。之后多个stream进行overlap
    // e.g. 一个stream处理: H2D, 另外几个流分别只处理kernel, 和D2H。之后所有stream进行overlap
    // e.g. 一个stream处理: H2D(局部), kernel(局部), D2H(局部)。之后所有stream进行overlap
    return 0;
}
