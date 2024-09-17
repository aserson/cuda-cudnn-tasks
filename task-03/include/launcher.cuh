#pragma once

#include <iostream>

#include <cuda_runtime.h>

#define CUDNN_CHECK_ERROR(err) { \
    if (err != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "CUDDNN Error: " << cudnnGetErrorString(err) << " at line " << __LINE__ << " in file " << __FILE__ << std::endl; \
        exit(1); \
    } \
}

#define CUDA_CHECK_ERROR(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << " in file " << __FILE__ << std::endl; \
        exit(1); \
    } \
}

template <typename KernelFunc, typename... TArgs>
void launch_kernel(unsigned int threads_per_block, unsigned int task_size, KernelFunc kernel, TArgs... args) {
    unsigned int blocks = (task_size % threads_per_block == 0) ? task_size / threads_per_block
        : task_size / threads_per_block + 1;

    kernel << <blocks, threads_per_block >> > (args...);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

template <typename KernelFunc, typename... TArgs>
unsigned int launch_shared_kernel(unsigned int threads_per_block, unsigned int task_size, 
                                    KernelFunc kernel, TArgs... args) {
    unsigned int blocks = (task_size + threads_per_block - 1) / threads_per_block;

    kernel<< <blocks, threads_per_block, threads_per_block * sizeof(float) >> > (args...);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    return blocks;
}