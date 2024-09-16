#pragma once

#include "kernels.cuh"

#include <cudnn.h>

class Filter {
    int dim, channels;
    int size, bytes;

    cudnnFilterDescriptor_t descriptor;

    float* h_filter;
    float* d_filter;

public:
    Filter(int filter_dim, int channels)
        : dim(filter_dim), channels(channels) {

        size = filter_dim * filter_dim * channels * channels;
        bytes = size * sizeof(float);

        CUDA_CHECK_ERROR(cudaMalloc(&d_filter, bytes));
        CUDA_CHECK_ERROR(cudaMallocHost(&h_filter, bytes, cudaHostAllocDefault));

        CUDNN_CHECK_ERROR(cudnnCreateFilterDescriptor(&descriptor));
        CUDNN_CHECK_ERROR(cudnnSetFilter4dDescriptor(descriptor,
            CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
            channels, channels,
            filter_dim, filter_dim));
    }

    ~Filter() {
        CUDNN_CHECK_ERROR(cudnnDestroyFilterDescriptor(descriptor));
        CUDA_CHECK_ERROR(cudaFree(d_filter));
        CUDA_CHECK_ERROR(cudaFreeHost(h_filter));
    }

    const cudnnFilterDescriptor_t& getDesctiptor() const {
        return descriptor;
    }

    float* getFilter() {
        return d_filter;
    }

    void load_filter(float* input_filter) {
        CUDA_CHECK_ERROR(cudaMemcpy(h_filter, input_filter, bytes, cudaMemcpyHostToHost));
        synchronizeWithDevice();
    }

    void generate_identity_filter() {
        std::memset(h_filter, 0x0, bytes);

        for (int j = 0; j < channels; ++j) {
            int block = j * channels + j;
            int block_size = dim * dim;

            int idx = block * block_size + (block_size - 1) / 2;

            h_filter[idx] = 1;
        }

        synchronizeWithDevice();
    }

    void generate_sharpen_filter() {
        if (dim == 3) {
            std::memset(h_filter, 0x0, bytes);
            int block_size = dim * dim;

            for (int ch = 0; ch < channels; ++ch) {
                int block = ch * channels + ch;

                h_filter[block * block_size + 1] = -1;
                h_filter[block * block_size + 3] = -1;
                h_filter[block * block_size + 4] = 5;
                h_filter[block * block_size + 5] = -1;
                h_filter[block * block_size + 7] = -1;
            }

            synchronizeWithDevice();
        }
        else {
            std::cout << "Filter dim not equal 3" << std::endl;
            std::cout << "Generating identity filter" << std::endl;

            generate_identity_filter();
        }
    }

    void generate_blure_filter() {
        std::memset(h_filter, 0x0, bytes);
        int block_size = dim * dim;

        float val = 1.f / static_cast<float>(block_size);

        for (int ch = 0; ch < channels; ++ch) {
            int block = ch * channels + ch;
             
            for (int i = 0; i < block_size; ++i)
                h_filter[block * block_size + i] = val;
        }

        synchronizeWithDevice();
    }

    void generate_gaussian_blure_filter(int size = 3) {
        if (size == 3 && size == dim) {
            std::memset(h_filter, 0x0, bytes);
            int block_size = dim * dim;

            float val = 1.f / 16.f;

            for (int ch = 0; ch < channels; ++ch) {
                int block = ch * channels + ch;

                h_filter[block * block_size + 0] = 1.f * val;
                h_filter[block * block_size + 1] = 2.f * val;
                h_filter[block * block_size + 2] = 1.f * val;
                h_filter[block * block_size + 3] = 2.f * val;
                h_filter[block * block_size + 4] = 4.f * val;
                h_filter[block * block_size + 5] = 2.f * val;
                h_filter[block * block_size + 6] = 1.f * val;
                h_filter[block * block_size + 7] = 2.f * val;
                h_filter[block * block_size + 8] = 1.f * val;
            }

            synchronizeWithDevice();
        }
        else if (size == 5 && size == dim) {
            std::memset(h_filter, 0x0, bytes);
            int block_size = dim * dim;

            float val = 1.f / 256.f;

            for (int ch = 0; ch < channels; ++ch) {
                int block = ch * channels + ch;

                h_filter[block * block_size + 0] = 1.f * val;
                h_filter[block * block_size + 1] = 4.f * val;
                h_filter[block * block_size + 2] = 6.f * val;
                h_filter[block * block_size + 3] = 4.f * val;
                h_filter[block * block_size + 4] = 1.f * val;
                h_filter[block * block_size + 5] = 4.f * val;
                h_filter[block * block_size + 6] = 16.f * val;
                h_filter[block * block_size + 7] = 24.f * val;
                h_filter[block * block_size + 8] = 16.f * val;
                h_filter[block * block_size + 9] = 4.f * val;
                h_filter[block * block_size + 10] = 6.f * val;
                h_filter[block * block_size + 11] = 24.f * val;
                h_filter[block * block_size + 12] = 36.f * val;
                h_filter[block * block_size + 13] = 24.f * val;
                h_filter[block * block_size + 14] = 6.f * val;
                h_filter[block * block_size + 15] = 4.f * val;
                h_filter[block * block_size + 16] = 16.f * val;
                h_filter[block * block_size + 17] = 26.f * val;
                h_filter[block * block_size + 18] = 16.f * val;
                h_filter[block * block_size + 19] = 4.f * val;
                h_filter[block * block_size + 20] = 1.f * val;
                h_filter[block * block_size + 21] = 4.f * val;
                h_filter[block * block_size + 22] = 6.f * val;
                h_filter[block * block_size + 23] = 4.f * val;
                h_filter[block * block_size + 24] = 1.f * val;
            }

            synchronizeWithDevice();
        }
        else {
            std::cout << "Dim = " << size << " not supported" << std::endl;
            std::cout << "Generating simple blur filter" << std::endl;

            generate_blure_filter();
        }
    }

    void generate_unsharp_masking_filter() {
        if (5 == dim) {
            std::memset(h_filter, 0x0, bytes);
            int block_size = dim * dim;

            float val = -1.f / 256.f;

            for (int ch = 0; ch < channels; ++ch) {
                int block = ch * channels + ch;

                h_filter[block * block_size + 0] = 1.f * val;
                h_filter[block * block_size + 1] = 4.f * val;
                h_filter[block * block_size + 2] = 6.f * val;
                h_filter[block * block_size + 3] = 4.f * val;
                h_filter[block * block_size + 4] = 1.f * val;
                h_filter[block * block_size + 5] = 4.f * val;
                h_filter[block * block_size + 6] = 16.f * val;
                h_filter[block * block_size + 7] = 24.f * val;
                h_filter[block * block_size + 8] = 16.f * val;
                h_filter[block * block_size + 9] = 4.f * val;
                h_filter[block * block_size + 10] = 6.f * val;
                h_filter[block * block_size + 11] = 24.f * val;
                h_filter[block * block_size + 12] = -476.f * val;
                h_filter[block * block_size + 13] = 24.f * val;
                h_filter[block * block_size + 14] = 6.f * val;
                h_filter[block * block_size + 15] = 4.f * val;
                h_filter[block * block_size + 16] = 16.f * val;
                h_filter[block * block_size + 17] = 26.f * val;
                h_filter[block * block_size + 18] = 16.f * val;
                h_filter[block * block_size + 19] = 4.f * val;
                h_filter[block * block_size + 20] = 1.f * val;
                h_filter[block * block_size + 21] = 4.f * val;
                h_filter[block * block_size + 22] = 6.f * val;
                h_filter[block * block_size + 23] = 4.f * val;
                h_filter[block * block_size + 24] = 1.f * val;
            }

            synchronizeWithDevice();
        }
        else {
            std::cout << "Filter dim not equal 5" << std::endl;
            std::cout << "Generating simple sharpen filter" << std::endl;

            generate_sharpen_filter();
        }
    }

    void generate_ridge_filter() {
        if (dim == 3) {
            std::memset(h_filter, 0x0, bytes);
            int block_size = dim * dim;

            for (int block = 0; block < channels * channels; ++block) {
                h_filter[block * block_size + 1] = -1;
                h_filter[block * block_size + 3] = -1;
                h_filter[block * block_size + 4] = 4;
                h_filter[block * block_size + 5] = -1;
                h_filter[block * block_size + 7] = -1;
            }

            synchronizeWithDevice();
        }
        else {
            std::cout << "Filter dim not equal 3" << std::endl;
            std::cout << "Generating identity filter" << std::endl;

            generate_identity_filter();
        }
    }

private:
    void synchronizeWithHost() {
        CUDA_CHECK_ERROR(cudaMemcpy(h_filter, d_filter, bytes, cudaMemcpyDeviceToHost));
    }

    void synchronizeWithDevice() {
        CUDA_CHECK_ERROR(cudaMemcpy(d_filter, h_filter, bytes, cudaMemcpyHostToDevice));
    }

};