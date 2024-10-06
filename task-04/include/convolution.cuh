#pragma once

#include "image.h"
#include <cudnn.h>
#include "filter.cuh"

#include <thread>
#include <atomic>

typedef float TData;

class Convolution {
    cudnnDataType_t type;

    cudnnHandle_t cudnn;

    Filter filter;

    cudnnConvolutionDescriptor_t conv_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;

    cudnnTensorDescriptor_t data_descriptor;

    cudnnConvolutionFwdAlgoPerf_t algo_pref;
    size_t f_workspace_bytes = 0;
    void* f_workspace = nullptr;

    cudnnConvolutionBwdFilterAlgoPerf_t algo_pref_bwd;
    size_t b_workspace_bytes = 0;
    void* b_workspace = nullptr;

    int channels = 0, height = 0, width = 0, numbers = 0;
    int pic_size = 0;
    int dim = 0;
    int size = 0, bytes = 0;

    std::vector<TData> tmp;
    TData *input = nullptr;
    TData *output = nullptr;
    TData *expected = nullptr;
    TData *output_grad = nullptr;
    TData *d_tmp = nullptr;

    // output = alpha × new_result + beta × previous_output
    TData alpha = 1.0f, beta = 0.0f, epsilon = 0.01f;

    TData rate_coef = 0.01f;

    int count = 0;
    std::atomic<bool> stop_flag;
public:
    Convolution(cudnnDataType_t type = CUDNN_DATA_FLOAT);
    ~Convolution();

    bool set_data_desc(const ImageBatch& input);
    bool set_filter_desc(int filter_dim, int filter_channels);

    bool set_convolution();
    bool set_convolution_bwd();

    bool set_input(const ImageBatch& input);
    bool set_expected(const ImageBatch& expected);
    bool set_output();
    bool set_output_grad();

    bool set_filter(FilterName name);

    bool apply_forward();
    bool apply_backward();
    bool gradient_descent(int max_cycles = 100000);

    bool get_output(ImageBatch& output);

private:
    TData* fromNHWCtoNCHW(const ImageBatch& h_input);

    TData find_loss_gradient();
    TData find_learning_rate();
    void apply_filter_gradient(TData learning_rate);
    void listen_for_commands();
};
