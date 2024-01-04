#include "timer.h"
#include "utils.h"

Timer::Timer() {
    _timeElasped = 0;
    _cStart = std::chrono::high_resolution_clock::now();
    _cStop  = std::chrono::high_resolution_clock::now();
    cudaEventCreate(&_gStart);
    cudaEventCreate(&_gStop);
}

Timer::~Timer() {
    cudaEventDestroy(_gStart);
    cudaEventDestroy(_gStop);
}

void Timer::start_gpu() {
    cudaEventRecord(_gStart, 0);
}

void Timer::stop_gpu() {
    cudaEventRecord(_gStop, 0);
}

void Timer::duration_gpu(std::string msg) {
    CUDA_CHECK(cudaEventSynchronize(_gStart));
    CUDA_CHECK(cudaEventSynchronize(_gStop));
    cudaEventElapsedTime(&_timeElasped, _gStart, _gStop);

    LOG("%-60s cost %.6lf ms", msg.c_str(), _timeElasped);
}