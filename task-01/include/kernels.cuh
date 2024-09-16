#pragma once

#include <iostream>

#include <cuda_runtime.h>

#define CUDNN_CHECK_ERROR(err) { \
    if (err != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "CUDDNN Error: " << cudnnGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

#define CUDA_CHECK_ERROR(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

template <typename KernelFunc, typename... TArgs>
void launch_kernel(dim3 blocks, dim3 threads, KernelFunc kernel, TArgs... args) {
    kernel<<<blocks, threads>>>(args...);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

template <typename... TArgs>
unsigned int launch_max_kernel(unsigned int threads_per_block, unsigned int task_size, TArgs... args) {
    unsigned int blocks = (task_size % threads_per_block == 0) ? task_size / threads_per_block
        : task_size / threads_per_block + 1;

    find_max <<<blocks, threads_per_block, threads_per_block  * sizeof(float) >> >(args...);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    return blocks;
}

__global__
void find_max(const float *input, int size, float* tmp) {
    extern __shared__ float shared_mem[];

    int thread_idx = threadIdx.x;
    int block_idx = blockIdx.x;
    int block_dim = blockDim.x;

    int global_idx = block_idx * block_dim + thread_idx;

    if (global_idx < size) {
        shared_mem[thread_idx] = input[global_idx];
    } else {
        shared_mem[thread_idx] = - FLT_MAX;
    }
    __syncthreads();

    for (int stride = block_dim / 2; stride > 0; stride >>= 1) {
        if (thread_idx < stride && global_idx + stride < size) {
            if (shared_mem[thread_idx] < shared_mem[thread_idx + stride])
                shared_mem[thread_idx] = shared_mem[thread_idx + stride];
        }
        __syncthreads();
    }

    if (thread_idx == 0) {
        tmp[block_idx] = shared_mem[0];
    }
}
