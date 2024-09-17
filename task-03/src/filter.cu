#include "filter.cuh"

#include <cstring>

#include "launcher.cuh"

Filter::~Filter() {
    if (filter != nullptr) CUDA_CHECK_ERROR(cudaFree(filter));
    if (grad != nullptr) CUDA_CHECK_ERROR(cudaFree(grad));
}

void Filter::set_filter(float* input_filter, int input_dim, int input_channels) {
    dim = input_dim;
    channels = input_channels;

    size = dim * dim * channels * channels;
    bytes = size * sizeof(float);

    CUDA_CHECK_ERROR(cudaMalloc(&filter, bytes));
    CUDA_CHECK_ERROR(cudaMemcpy(filter, input_filter, bytes, cudaMemcpyHostToDevice));
}

void Filter::set_grad() {
    CUDA_CHECK_ERROR(cudaMalloc(&grad, bytes));
    CUDA_CHECK_ERROR(cudaMemset(grad, 0x0, bytes));
}

bool Filter::generate_filter(FilterName name, int generated_dim, int generated_channels) {
    dim = generated_dim;
    channels = generated_channels;

    size = dim * dim * channels * channels;
    bytes = size * sizeof(float);

    tmp.resize(size, 0.f);

    switch (name) {
        case ZERO_FILTER:
            break;
        case IDENTITY_FILTER:
            generate_identity_filter();
            break;
        case SHARPEN_FILTER:
            generate_sharpen_filter();
            break;
        case BLURE_FILTER:
            generate_blure_filter();
            break;
        case GAUSSIAN_BLURE_FILTER:
            generate_gaussian_blure_filter();
            break;
        case UNSHARP_MASKING_FILTER:
            generate_unsharp_masking_filter();
            break;
        case RIDGE_FILTER:
            generate_ridge_filter();
            break;
        case WEEK_RIDGE_FILTER:
            generate_week_ridge_filter();
            break;
        default:
            return false;
    }

    CUDA_CHECK_ERROR(cudaMalloc(&filter, bytes));
    CUDA_CHECK_ERROR(cudaMemcpy(filter, tmp.data(), bytes, cudaMemcpyHostToDevice));

    return true;
}

void Filter::generate_identity_filter() {
    for (int j = 0; j < channels; ++j) {
        int block = j * channels + j;
        int block_size = dim * dim;

        int idx = block * block_size + (block_size - 1) / 2;

        tmp[idx] = 1;
    }
}

void Filter::generate_sharpen_filter() {
    if (dim == 3) {
        int block_size = dim * dim;

        for (int ch = 0; ch < channels; ++ch) {
            int block = ch * channels + ch;

            tmp[block * block_size + 1] = -1;
            tmp[block * block_size + 3] = -1;
            tmp[block * block_size + 4] = 5;
            tmp[block * block_size + 5] = -1;
            tmp[block * block_size + 7] = -1;
        }
    }
    else {
        std::cout << "Filter dim not equal 3" << std::endl;
        std::cout << "Generating identity filter" << std::endl;

        generate_identity_filter();
    }
}

void Filter::generate_blure_filter() {
    int block_size = dim * dim;

    float val = 1.f / static_cast<float>(block_size);

    for (int ch = 0; ch < channels; ++ch) {
        int block = ch * channels + ch;

        for (int i = 0; i < block_size; ++i)
            tmp[block * block_size + i] = val;
    }
}

void Filter::generate_gaussian_blure_filter() {
    if (dim == 3) {
        int block_size = dim * dim;

        float val = 1.f / 16.f;

        for (int ch = 0; ch < channels; ++ch) {
            int block = ch * channels + ch;

            tmp[block * block_size + 0] = 1.f * val;
            tmp[block * block_size + 1] = 2.f * val;
            tmp[block * block_size + 2] = 1.f * val;
            tmp[block * block_size + 3] = 2.f * val;
            tmp[block * block_size + 4] = 4.f * val;
            tmp[block * block_size + 5] = 2.f * val;
            tmp[block * block_size + 6] = 1.f * val;
            tmp[block * block_size + 7] = 2.f * val;
            tmp[block * block_size + 8] = 1.f * val;
        }
    }
    else if (dim == 5) {
        int block_size = dim * dim;

        float val = 1.f / 256.f;

        for (int ch = 0; ch < channels; ++ch) {
            int block = ch * channels + ch;

            tmp[block * block_size + 0] = 1.f * val;
            tmp[block * block_size + 1] = 4.f * val;
            tmp[block * block_size + 2] = 6.f * val;
            tmp[block * block_size + 3] = 4.f * val;
            tmp[block * block_size + 4] = 1.f * val;
            tmp[block * block_size + 5] = 4.f * val;
            tmp[block * block_size + 6] = 16.f * val;
            tmp[block * block_size + 7] = 24.f * val;
            tmp[block * block_size + 8] = 16.f * val;
            tmp[block * block_size + 9] = 4.f * val;
            tmp[block * block_size + 10] = 6.f * val;
            tmp[block * block_size + 11] = 24.f * val;
            tmp[block * block_size + 12] = 36.f * val;
            tmp[block * block_size + 13] = 24.f * val;
            tmp[block * block_size + 14] = 6.f * val;
            tmp[block * block_size + 15] = 4.f * val;
            tmp[block * block_size + 16] = 16.f * val;
            tmp[block * block_size + 17] = 26.f * val;
            tmp[block * block_size + 18] = 16.f * val;
            tmp[block * block_size + 19] = 4.f * val;
            tmp[block * block_size + 20] = 1.f * val;
            tmp[block * block_size + 21] = 4.f * val;
            tmp[block * block_size + 22] = 6.f * val;
            tmp[block * block_size + 23] = 4.f * val;
            tmp[block * block_size + 24] = 1.f * val;
        }
    }
    else {
        std::cout << "Dim = " << dim << " not supported" << std::endl;
        std::cout << "Generating simple blur filter" << std::endl;

        generate_blure_filter();
    }
}

void Filter::generate_unsharp_masking_filter() {
    if (dim == 5) {
        int block_size = dim * dim;

        float val = -1.f / 256.f;

        for (int ch = 0; ch < channels; ++ch) {
            int block = ch * channels + ch;

            tmp[block * block_size + 0] = 1.f * val;
            tmp[block * block_size + 1] = 4.f * val;
            tmp[block * block_size + 2] = 6.f * val;
            tmp[block * block_size + 3] = 4.f * val;
            tmp[block * block_size + 4] = 1.f * val;
            tmp[block * block_size + 5] = 4.f * val;
            tmp[block * block_size + 6] = 16.f * val;
            tmp[block * block_size + 7] = 24.f * val;
            tmp[block * block_size + 8] = 16.f * val;
            tmp[block * block_size + 9] = 4.f * val;
            tmp[block * block_size + 10] = 6.f * val;
            tmp[block * block_size + 11] = 24.f * val;
            tmp[block * block_size + 12] = -476.f * val;
            tmp[block * block_size + 13] = 24.f * val;
            tmp[block * block_size + 14] = 6.f * val;
            tmp[block * block_size + 15] = 4.f * val;
            tmp[block * block_size + 16] = 16.f * val;
            tmp[block * block_size + 17] = 26.f * val;
            tmp[block * block_size + 18] = 16.f * val;
            tmp[block * block_size + 19] = 4.f * val;
            tmp[block * block_size + 20] = 1.f * val;
            tmp[block * block_size + 21] = 4.f * val;
            tmp[block * block_size + 22] = 6.f * val;
            tmp[block * block_size + 23] = 4.f * val;
            tmp[block * block_size + 24] = 1.f * val;
        }
    }
    else {
        std::cout << "Filter dim not equal 5" << std::endl;
        std::cout << "Generating simple sharpen filter" << std::endl;

        generate_sharpen_filter();
    }
}

void Filter::generate_ridge_filter() {
    if (dim == 3) {
        int block_size = dim * dim;

        for (int block = 0; block < channels * channels; ++block) {
            tmp[block * block_size + 1] = -1;
            tmp[block * block_size + 3] = -1;
            tmp[block * block_size + 4] = 4;
            tmp[block * block_size + 5] = -1;
            tmp[block * block_size + 7] = -1;
        }
    }
    else {
        std::cout << "Filter dim not equal 3" << std::endl;
        std::cout << "Generating identity filter" << std::endl;

        generate_identity_filter();
    }
}

void Filter::generate_week_ridge_filter() {
    if (dim == 3) {
        int block_size = dim * dim;

        for (int block = 0; block < channels * channels; ++block) {
            tmp[block * block_size + 1] = -0.5;
            tmp[block * block_size + 3] = -0.5;
            tmp[block * block_size + 4] = 2;
            tmp[block * block_size + 5] = -0.5;
            tmp[block * block_size + 7] = -0.5;
        }
    }
    else {
        std::cout << "Filter dim not equal 3" << std::endl;
        std::cout << "Generating identity filter" << std::endl;

        generate_identity_filter();
    }
}