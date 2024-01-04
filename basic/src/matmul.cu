#include <cstdio>
#include <random>
#include <math.h>
#include <cuda_runtime.h>
#include "utils.h"
#include "timer.h"
#include "matmul.h"

__global__ void MatmulKernel(float *M_device, float *N_device, float *P_device, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float P_element = 0;

    for (int k = 0; k < width; k++) {
        float M_element = *(M_device + y * width + k);
        float N_element = *(N_device + k * width + x);
        P_element += M_element * N_element;
    }
    *(P_device + y * width + x) = P_element;
}

__global__ void MatmulSharedStaticKernel(float *M_device, float *N_device, float *P_device, int width) {
    __shared__ float M_deviceShared[BLOCKSIZE][BLOCKSIZE];
    __shared__ float N_deviceShared[BLOCKSIZE][BLOCKSIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float P_element = 0;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    for (int m = 0; m < width / BLOCKSIZE; m++) {
        M_deviceShared[ty][tx] = M_device[y * width + (m * BLOCKSIZE + tx)];
        N_deviceShared[ty][tx] = N_device[(m * BLOCKSIZE + ty) * width + x];
        __syncthreads();
        
        for (int k = 0; k < BLOCKSIZE; k++) {
            P_element += M_deviceShared[ty][k] * N_deviceShared[k][tx];
        }
        __syncthreads();
    }
    *(P_device + y * width + x) = P_element;
}

__global__ void MatmulSharedDynamicKernel(float *M_device, float *N_device, float *P_device, int width, int blockSize) {
    extern __shared__ float deviceShared[];
    int stride = blockSize * blockSize;
    /*
        对于x和y, 根据blockID, tile大小和threadID进行索引
    */
    int x = blockIdx.x * blockSize + threadIdx.x;
    int y = blockIdx.y * blockSize + threadIdx.y;

    float P_element = 0.0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    /* 对于每一个P的元素，我们只需要循环遍历width / tile_width 次就okay了 */
    for (int m = 0; m < width / blockSize; m++) {
        deviceShared[ty * blockSize + tx] = M_device[y * width + (m * blockSize + tx)];
        deviceShared[stride + (ty * blockSize + tx)] = N_device[(m * blockSize + ty) * width + x];
        __syncthreads();

        for (int k = 0; k < blockSize; k++) {
            P_element += deviceShared[ty * blockSize + k] * deviceShared[stride + (k * blockSize + tx)];
        }
        __syncthreads();
    }

    if (y < width && x < width) {
        P_device[y * width + x] = P_element;
    }
}

__global__ void MatmulAtomicKernel(float *dM1, float *dM2, float *dP) {
    int colIdx = blockIdx.x;
    int rowIdx = blockIdx.y;
    int width = gridDim.y;
    int k = threadIdx.x;

    atomicAdd(&dP[rowIdx * width + colIdx], dM1[rowIdx * width + k] * dM2[(rowIdx + k) * width + colIdx]);
}

__global__ void MatmulSharedStaticConflicktKernel(float *M_device, float *N_device, float *P_device, int width) {
    // 添加 padding 防止bank conflict发生
    __shared__ float M_deviceShared[BLOCKSIZE][BLOCKSIZE + 1];
    __shared__ float N_deviceShared[BLOCKSIZE][BLOCKSIZE + 1];
    /*
        对于x和y, 根据blockID, tile大小和threadID进行索引
    */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float P_element = 0.0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    /* 对于每一个P的元素，我们只需要循环遍历width / tile_width 次就okay了，这里有点绕，画图理解一下*/
    for (int m = 0; m < width / BLOCKSIZE; m++) {
        // 实现 bank conflict 
        M_deviceShared[tx][ty] = M_device[x * width + (m * BLOCKSIZE + ty)];
        N_deviceShared[tx][ty] = N_device[(m * BLOCKSIZE + tx) * width + y];
        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; k++) {
            P_element += M_deviceShared[tx][k] * N_deviceShared[k][ty];
        }
        __syncthreads();
    }
    // 列优先
    P_device[x * width + y] = P_element;
}

__global__ void MatmulSharedDynamicConflictKernel(float *M_device, float *N_device, float *P_device, int width, int blockSize) {
    /*
        声明动态共享变量的时候需要加extern，同时需要是一维的
        注意这里有个坑, 不能够像这样定义：
            __shared__ float M_deviceShared[];
            __shared__ float N_deviceShared[];
        因为在cuda中定义动态共享变量的话，无论定义多少个他们的地址都是一样的。
        所以如果想要像上面这样使用的话，需要用两个指针分别指向shared memory的不同位置才行
    */

    extern __shared__ float deviceShared[];
    int stride = (blockSize + 1) * blockSize;
    /*
        对于x和y, 根据blockID, tile大小和threadID进行索引
    */
    int x = blockIdx.x * blockSize + threadIdx.x;
    int y = blockIdx.y * blockSize + threadIdx.y;

    float P_element = 0.0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    /* 对于每一个P的元素，我们只需要循环遍历width / tile_width 次就okay了 */
    for (int m = 0; m < width / blockSize; m++) {
        deviceShared[tx * (blockSize + 1) + ty] = M_device[x * width + (m * blockSize + ty)];
        deviceShared[stride + (tx * (blockSize + 1) + ty)] = N_device[(m * blockSize + tx) * width + y];
        __syncthreads();

        for (int k = 0; k < blockSize; k++) {
            P_element += deviceShared[tx * (blockSize + 1) + k] * deviceShared[stride + (k * (blockSize + 1) + ty)];
        }
        __syncthreads();
    }

    if (y < width && x < width) {
        P_device[x * width + y] = P_element;
    }
}

void initMatrix(float *data, int size, int low, int high, int seed) {
    srand(seed);
    for (int i = 0; i < size; i++) {
        *(data + i) = float(rand()) * float(high - low) / RAND_MAX;
    }
}

void initMatrixSigned(float *data, int size, float low, float high, int seed) {
    srand(seed);
    for (int i = 0; i < size; i++) {
        data[i] = float(rand()) * float(high - low) / RAND_MAX;
        if (low >= 0) {
            data[i] -= low;
        } else {
            data[i] += low;
        }
    }
}

void printMat(float *data, int size)  {
    for (int i = 0; i < size; i++) {
        printf("%.8lf", *(data +i));
        if (i != size - 1) {
            printf(", ");
        } else {
            printf("\n");
        }
    }
}

void compareMat(float *h_data, float *d_data, int size) {
    double precision = 1.8E-4;
    bool error = false;
    for (int i = 0; i < size; i++) {
        if (std::abs(*(h_data + i) - *(d_data +i) > precision)) {
            error = true;
            printf("cpu: %.8lf, gpu: %.8lf\n", *(h_data + i), *(d_data +i));
            break;
        }
    }
    if (error) {
        printf("Matmul result is different\n");
    } else {
        printf("Matmul result is same, precision is 1.0E-4\n");
    }
}

void MatmulOnHost(float *M, float *N, float *P, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0;
            for (int k = 0; k < width; k++) {
                float a = *(M + i * width + k);
                float b = *(N + k * width + j);
                sum += a * b;
            }
            *(P + i * width + j) = sum;
        }
    }
}

void MatmulOnDevice(float *M_host, float *N_host, float *P_host, int width, int blockSize) {
    int size = width * width * sizeof(float);

    float *M_device;
    float *N_device;
    CUDA_CHECK(cudaMalloc(&M_device, size));
    CUDA_CHECK(cudaMalloc(&N_device, size));

    cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice);

    float *P_device;
    cudaMalloc(&P_device, size);

    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(width / blockSize, width / blockSize);
    MatmulKernel<<<dimGrid, dimBlock>>> (M_device, N_device, P_device, width);
    
    cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost);

    LAST_KERNEL_CHECK();
    cudaDeviceSynchronize();

    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);
}

void MatmulSharedOnDevice(float *M_host, float *N_host, float *P_host, int width, int blockSize, bool staticMem) {
    int size = width * width * sizeof(float);
    long int sMemSize = blockSize * blockSize * sizeof(float) * 2;

    float *M_device;
    float *N_device;
    CUDA_CHECK(cudaMalloc((void **) &M_device, size));
    CUDA_CHECK(cudaMalloc((void **) &N_device, size));

    CUDA_CHECK(cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice));

    float *P_device;
    CUDA_CHECK(cudaMalloc((void **) &P_device, size));

    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(width / blockSize, width / blockSize);
    if (staticMem) {
        MatmulSharedStaticKernel <<<dimGrid, dimBlock>>>(M_device, N_device, P_device, width);
    } else {
        MatmulSharedDynamicKernel <<<dimGrid, dimBlock, sMemSize, nullptr>>>(M_device, N_device, P_device, width, blockSize);
    }
    
    CUDA_CHECK(cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    LAST_KERNEL_CHECK();

    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);
}


void MatmulAtomicAddOnDevice(float *M_host, float *N_host, float *P_host, int width) {

    /* 设置矩阵大小 */
    int size = width * width * sizeof(float);

    /* 分配M, N在GPU上的空间*/
    float *M_device;
    float *N_device;
    CUDA_CHECK(cudaMalloc((void **) &M_device, size));
    CUDA_CHECK(cudaMalloc((void **) &N_device, size));

    /* 分配M, N拷贝到GPU上*/
    CUDA_CHECK(cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice));

    /* 分配P在GPU上的空间*/
    float *P_device;
    CUDA_CHECK(cudaMalloc((void **) &P_device, size));;

    dim3 dimGrid(width, width);
    dim3 dimBlock(width);
    MatmulAtomicKernel<<<dimGrid, dimBlock>>>(M_device, N_device, P_device);

    /* 将结果从device拷贝回host*/
    CUDA_CHECK(cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* 注意要在synchronization结束之后排查kernel的错误 */
    LAST_KERNEL_CHECK();

    /* Free */
    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);
}

void MatmulSharedConflictOnDevice(float *M_host, float *N_host, float *P_host, int width, int blockSize, bool staticMem) {
    /* 设置矩阵大小 */
    int size = width * width * sizeof(float);
    long int sMemSize = blockSize * blockSize * sizeof(float) * 2;

    /* 分配M, N在GPU上的空间*/
    float *M_device;
    float *N_device;
    CUDA_CHECK(cudaMalloc((void **) &M_device, size));
    CUDA_CHECK(cudaMalloc((void **) &N_device, size));

    /* 分配M, N拷贝到GPU上*/
    CUDA_CHECK(cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_device, N_host, size, cudaMemcpyHostToDevice));

    /* 分配P在GPU上的空间*/
    float *P_device;
    CUDA_CHECK(cudaMalloc((void **) &P_device, size));;

    /* 调用kernel来进行matmul计算, 在这个例子中我们用的方案是：使用一个grid，一个grid里有width*width个线程 */
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(width / blockSize, width / blockSize);
    if (staticMem) {
        MatmulSharedStaticConflicktKernel <<<dimGrid, dimBlock>>>(M_device, N_device, P_device, width);
    } else {
        MatmulSharedDynamicConflictKernel <<<dimGrid, dimBlock, sMemSize, nullptr>>>(M_device, N_device, P_device, width, blockSize);
    }

    /* 将结果从device拷贝回host*/
    CUDA_CHECK(cudaMemcpy(P_host, P_device, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* 注意要在synchronization结束之后排查kernel的错误 */
    LAST_KERNEL_CHECK();

    /* Free */
    cudaFree(P_device);
    cudaFree(N_device);
    cudaFree(M_device);
}