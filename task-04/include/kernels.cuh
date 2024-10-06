#pragma once

#include <cuda_runtime.h>

__global__ 
void find_gradient(unsigned int task_size, float* computed, float* expected, float* gradient) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < task_size)
        gradient[idx] =  computed[idx] - expected[idx];
}

__global__
void find_sum(float* input, int size, float* tmp) {
    extern __shared__ float shared_mem[];

    int thread_idx = threadIdx.x;
    int block_idx = blockIdx.x;
    int block_dim = blockDim.x;

    int global_idx = block_idx * block_dim + thread_idx;

    shared_mem[thread_idx] = (global_idx < size) ? fabs(input[global_idx]) : 0.0f;
    __syncthreads();

    for (int stride = block_dim / 2; stride > 0; stride >>= 1) {
        if (thread_idx < stride) {
            shared_mem[thread_idx] += shared_mem[thread_idx + stride];
        }
        __syncthreads();
    }

    if (thread_idx == 0) {
        tmp[block_idx] = shared_mem[0];
    }
}

__global__
void find_max(float* input, int size, float* tmp) {
    extern __shared__ float shared_mem[];

    int thread_idx = threadIdx.x;
    int block_idx = blockIdx.x;
    int block_dim = blockDim.x;

    int global_idx = block_idx * block_dim + thread_idx;

    shared_mem[thread_idx] = (global_idx < size) ? fabs(input[global_idx]) : 0.0f;
    __syncthreads();

    for (int stride = block_dim / 2; stride > 0; stride >>= 1) {
        if (thread_idx < stride) {
            if (shared_mem[thread_idx] < shared_mem[thread_idx + stride])
                shared_mem[thread_idx] = shared_mem[thread_idx + stride];
        }
        __syncthreads();
    }

    if (thread_idx == 0) {
        tmp[block_idx] = shared_mem[0];
    }
}

__global__
void apply_gradient(unsigned int task_size, float* filter, float* gradient, float learning_rate) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < task_size)
        filter[idx] = filter[idx] - learning_rate * gradient[idx];
}
