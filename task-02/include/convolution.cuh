#pragma once

#include <cudnn.h>
#include "filter.cuh"

class Convolution {
    cudnnHandle_t cudnn;

    Filter* filter = nullptr;

    cudnnConvolutionDescriptor_t conv_descriptor;
    cudnnConvolutionFwdAlgoPerf_t algoPref;

    cudnnPoolingDescriptor_t pooling_descriptor;

    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t c_output_descriptor;
    cudnnTensorDescriptor_t p_output_descriptor;

    int channels;
    int c_output_number, c_output_channels, c_output_height, c_output_width;
    int p_output_number, p_output_channels, p_output_height, p_output_width;

    int input_size;
    int c_output_size;
    int p_output_size;

    size_t c_workspace_bytes = 0;
    void* c_workspace = nullptr;

    std::vector<float> tmp;
    float* input = nullptr, *c_output = nullptr, *p_output = nullptr;

    bool conv_is_set = false;
    bool pool_is_set = false;

    bool convolved = false;
    bool pooled = false;

    // output = alpha × new_result + beta × previous_output
    float alpha = 1.0f, beta = 0.0f;
public:
    Convolution(int channels);
    ~Convolution();

    bool set_input(const uint8_t* buffer, int input_height, int input_width, int input_channels);

    bool set_filter(int dim, int filter_channels);
    bool set_convolution();
    bool set_pooling(int height, int width, int vertical_stride, int horizontal_stride);

    bool apply_convolution();
    bool apply_pooling();

    int get_convolution_output_size() { return c_output_size; }
    int get_convolution_output_height() { return c_output_height; }
    int get_convolution_output_width() { return c_output_width; }
    int get_convolution_output_channels() { return c_output_channels; }

    int get_pooling_output_size() { return p_output_size; }
    int get_pooling_output_height() { return p_output_height; }
    int get_pooling_output_width() { return p_output_width; }
    int get_pooling_output_channels() { return p_output_channels; }

    bool get_convolution_output(uint8_t* buffer, int buffer_size);
    bool get_pooling_output(uint8_t* buffer, int buffer_size);

private:
    void create_descriptors();
};
