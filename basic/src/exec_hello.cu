#include <stdio.h>
#include <stdlib.h> 

__global__ void print_from_gpu(void) {
	printf("Hello CUDA! from thread [%d,%d] \
		From device\n", threadIdx.x, blockIdx.x); 
}

int main(void) {  
	printf("Hello CUDA from host!\n"); 
	print_from_gpu<<<1,1>>>();
	cudaDeviceSynchronize(); // 停止 CPU 端线程的执行，直到 GPU 端完成之前 CUDA 的任务
return 0; 
}

