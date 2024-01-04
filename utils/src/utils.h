#ifndef _UTILS_H_
#define _UTILS_H_

#include <cuda_runtime.h>
#include <system_error>
#include <stdarg.h>
#include <math.h>
#include <random>
#include <stdio.h>

#define CUDA_CHECK(call)            __cudaCheck(call, __FILE__, __LINE__)
#define LAST_KERNEL_CHECK(call)     __kernelCheck(__FILE__, __LINE__)       
#define LOG(...)                    __log_info(__VA_ARGS__)

#define BLOCKSIZE                   16                                   


inline static void __cudaCheck(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {                                            
        std::printf("ERROR: %s:%d, ", file, line);                      
        std::printf("code:%s, reason:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));  
        exit(1);                                                           
    }     
}

inline static void __kernelCheck(const char* file, const int line) {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {                                            
        std::printf("ERROR: %s:%d, ", file, line);                      
        std::printf("code:%s, reason:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));  
        exit(1);                                                           
    }    
}

inline static void __log_info(const char* format, ...) {
    char msg[1000];
    va_list args;
    va_start(args, format);

    vsnprintf(msg, sizeof(msg), format, args);
     
    fprintf(stdout, "%s\n", msg);
    va_end(args);
}

#endif