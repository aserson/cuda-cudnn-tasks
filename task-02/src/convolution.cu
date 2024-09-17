#include <iostream>
#include <fstream>
#include <filesystem>
#include <cudnn.h>

#include "convolution.cuh"
#include "launcher.cuh"

Convolution::Convolution(int channels) : channels(channels) {
    create_descriptors();
}

Convolution::~Convolution() {
    // Очистка ресурсов
    CUDNN_CHECK_ERROR(cudnnDestroyTensorDescriptor(input_descriptor));
    CUDNN_CHECK_ERROR(cudnnDestroyTensorDescriptor(c_output_descriptor));
    CUDNN_CHECK_ERROR(cudnnDestroyTensorDescriptor(p_output_descriptor));

    CUDNN_CHECK_ERROR(cudnnDestroyConvolutionDescriptor(conv_descriptor));
    CUDNN_CHECK_ERROR(cudnnDestroyPoolingDescriptor(pooling_descriptor));

    CUDNN_CHECK_ERROR(cudnnDestroy(cudnn));

    if (c_workspace != nullptr) CUDA_CHECK_ERROR(cudaFree(c_workspace));

    if (input != nullptr) CUDA_CHECK_ERROR(cudaFree(input));
    if (c_output != nullptr) CUDA_CHECK_ERROR(cudaFree(c_output));
    if (p_output != nullptr) CUDA_CHECK_ERROR(cudaFree(p_output));

    if (filter != nullptr) delete filter;
}

bool Convolution::set_input(const uint8_t* buffer, int input_height, int input_width, int input_channels) {
    if (input_channels != channels) {
        std::cout << "Input dimentional not equal initial dimentional" << std::endl;
        return false;
    }

    input_size = input_height * input_width * input_channels;

    tmp.resize(input_size);

    for (int i = 0; i < input_size; ++i) {
        tmp[i] = static_cast<float>(buffer[i]);
    }

    CUDA_CHECK_ERROR(cudaMalloc(&input, input_size * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMemcpy(input, tmp.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));

    CUDNN_CHECK_ERROR(cudnnSetTensor4dDescriptor(input_descriptor,
        CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
        1, input_channels, input_height, input_width));

    return true;
}

bool Convolution::set_filter(int dim, int filter_channels) {
    if (filter_channels != channels) {
        std::cout << "Filter dimentional not equal initial dimentional" << std::endl;
        return false;
    }

    filter = new Filter(dim, filter_channels);
    filter->generate_ridge_filter();

    return true;
}

bool Convolution::set_convolution() {
    conv_is_set = false;

    if (filter == nullptr || input == nullptr) {
        std::cout << "Set filter and input firstly" << std::endl;
        return conv_is_set;
    }

    int padding = (filter->getDim() - 1) / 2;
    CUDNN_CHECK_ERROR(cudnnSetConvolution2dDescriptor(conv_descriptor,
        padding, padding, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    CUDNN_CHECK_ERROR(cudnnGetConvolution2dForwardOutputDim(conv_descriptor,
        input_descriptor, filter->getDesctiptor(),
        &c_output_number, &c_output_channels, &c_output_height, &c_output_width));

    CUDNN_CHECK_ERROR(cudnnSetTensor4dDescriptor(c_output_descriptor,
        CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
        c_output_number, c_output_channels, c_output_height, c_output_width));

    // Allocation Output Data
    c_output_size = c_output_height * c_output_width * c_output_channels;
    CUDA_CHECK_ERROR(cudaMalloc(&c_output, c_output_size * sizeof(float)));

    if (tmp.size() < c_output_size)
        tmp.resize(c_output_size);

    // Creation best algoritm
    int alfoSize = 1;
    CUDNN_CHECK_ERROR(cudnnFindConvolutionForwardAlgorithm(cudnn,
        input_descriptor, filter->getDesctiptor(),
        conv_descriptor, c_output_descriptor,
        alfoSize, &alfoSize, &algoPref));

    // Allocation Workspace
    CUDNN_CHECK_ERROR(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
        input_descriptor, filter->getDesctiptor(), conv_descriptor,
        c_output_descriptor, algoPref.algo, &c_workspace_bytes));
    CUDA_CHECK_ERROR(cudaMalloc(&c_workspace, c_workspace_bytes));

    conv_is_set = true;
    return conv_is_set;
}

bool Convolution::set_pooling(int height, int width, int vertical_stride, int horizontal_stride) {
    pool_is_set = false;

    if (!conv_is_set) {
        std::cout << "Set convolution firstly" << std::endl;
        return pool_is_set;
    }

    CUDNN_CHECK_ERROR(cudnnSetPooling2dDescriptor(pooling_descriptor,
        CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
        height, width, 0, 0,
        vertical_stride, horizontal_stride));

    CUDNN_CHECK_ERROR(cudnnGetPooling2dForwardOutputDim(
        pooling_descriptor, c_output_descriptor,
        &p_output_number, &p_output_channels, &p_output_height, &p_output_width));

    CUDNN_CHECK_ERROR(cudnnSetTensor4dDescriptor(p_output_descriptor,
        CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
        p_output_number, p_output_channels, p_output_height, p_output_width));

    // Allocation Output Data
    p_output_size = p_output_height * p_output_width * p_output_channels;
    CUDA_CHECK_ERROR(cudaMalloc(&p_output, p_output_size * sizeof(float)));

    if (tmp.size() < p_output_size)
        tmp.resize(p_output_size);

    pool_is_set = true;
    return pool_is_set;
}

bool Convolution::apply_convolution() {
    convolved = false;

    if (!conv_is_set) {
        std::cout << "Set convolution firstly" << std::endl;
        return convolved;
    }

    CUDNN_CHECK_ERROR(cudnnConvolutionForward(cudnn,
        &alpha, input_descriptor, input,
        filter->getDesctiptor(), filter->getFilter(),
        conv_descriptor, algoPref.algo,
        c_workspace, c_workspace_bytes, &beta,
        c_output_descriptor, c_output));

    convolved = true;
    return convolved;
}

bool Convolution::apply_pooling() {
    pooled = false;

    if (!pool_is_set) {
        std::cout << "Set pooling firstly" << std::endl;
        return pooled;
    }

    if (!convolved) {
        apply_convolution();
    }

    CUDNN_CHECK_ERROR(cudnnPoolingForward(cudnn, pooling_descriptor, 
        &alpha, c_output_descriptor, c_output, 
        &beta, p_output_descriptor, p_output));

    pooled = true;
    return pooled;
}

bool Convolution::get_convolution_output(uint8_t* buffer, int buffer_size) {
    if (!conv_is_set)
        set_convolution();

    if (buffer_size != c_output_size) {
        std::cout << "Output Buffer size should be allocated with " << c_output_size << " elements" << std::endl;
        return false;
    }

    if (!convolved) {
        apply_convolution();
    }

    CUDA_CHECK_ERROR(cudaMemcpy(tmp.data(), c_output, c_output_size * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < c_output_size; ++i) {
        float val = std::min(std::max(tmp[i], 0.f), 255.f);
        buffer[i] = static_cast<uint8_t>(val);
    }

    return true;
}

bool Convolution::get_pooling_output(uint8_t* buffer, int buffer_size) {
    if (!pool_is_set) {
        std::cout << "Set pooling firstly" << std::endl;
        return false;
    }

    if (buffer_size != p_output_size) {
        std::cout << "Output Buffer size should be allocated with " << p_output_size << " elements" << std::endl;
        return false;
    }

    if (!pooled) {
        apply_pooling();
    }

    CUDA_CHECK_ERROR(cudaMemcpy(tmp.data(), p_output, p_output_size * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < p_output_size; ++i) {
        float val = std::min(std::max(tmp[i], 0.f), 255.f);

        buffer[i] = static_cast<uint8_t>(val);
    }

    return true;
}

void Convolution::create_descriptors() {
    CUDNN_CHECK_ERROR(cudnnCreate(&cudnn));

    // Создание дескрипторов памяти
    CUDNN_CHECK_ERROR(cudnnCreateTensorDescriptor(&input_descriptor));
    CUDNN_CHECK_ERROR(cudnnCreateTensorDescriptor(&c_output_descriptor));
    CUDNN_CHECK_ERROR(cudnnCreateTensorDescriptor(&p_output_descriptor));

    // Создание дескриптора под свёртку
    CUDNN_CHECK_ERROR(cudnnCreateConvolutionDescriptor(&conv_descriptor));

    // Создание дескриптора под пулинг
    CUDNN_CHECK_ERROR(cudnnCreatePoolingDescriptor(&pooling_descriptor));
}
