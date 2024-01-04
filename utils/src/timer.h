#ifndef _TIMER_H_
#define _TIMER_H_

#include <cstdio>
#include <ratio>
#include <string>
#include <chrono>
#include <cuda_runtime.h>


class Timer {
public:
    // std::ratio<num, den>（约分后的有理数类型）
    using s = std::ratio<1, 1>;
    using ms = std::ratio<1, 1000>;
    using us = std::ratio<1, 1000000>;
    using ns = std::ratio<1, 1000000000>;

    Timer();
    ~Timer();

    void start_cpu() {_cStart = std::chrono::high_resolution_clock::now();}
    void stop_cpu()  {_cStop  = std::chrono::high_resolution_clock::now();}

    void start_gpu();
    void stop_gpu();

    template <typename span>
    void duration_cpu(std::string msg);

    void duration_gpu(std::string msg);

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> _cStart;
    std::chrono::time_point<std::chrono::high_resolution_clock> _cStop;
    cudaEvent_t _gStart;
    cudaEvent_t _gStop;
    float _timeElasped;
};

#endif


template <typename span>
void Timer::duration_cpu(std::string msg) {
    std::string str;
    char fMsg[100];
    std::sprintf(fMsg, "%-60s", msg.c_str());
    if (std::is_same<span, s>::value) { str = " s";}
    else if (std::is_same<span, ms>::value) { str = " ms";}
    else if (std::is_same<span, us>::value) { str = " us";}
    else if (std::is_same<span, ns>::value) { str = " ns";}

    std::chrono::duration<double, span> time = _cStop - _cStart;
    LOG("%s cost %.6lf %s", fMsg, time.count(), str.c_str());
}