#include <iostream>
#include <fstream>
#include <filesystem>
#include <cudnn.h>

#include "convolution.h"
#include "kernels.cuh"
#include "filter.cuh"

class Convolution {
    cudnnHandle_t cudnn;

    Filter filter;

    cudnnTensorDescriptor_t input_descriptor;
    cudnnConvolutionDescriptor_t conv_descriptor;
    cudnnTensorDescriptor_t output_descriptor;

    int input_number, input_height, input_width, input_channels;
    int output_number, output_channels, output_height, output_width;

    int input_size;
    int output_size;

    int filter_dim;
    int filter_size;

    std::vector<uint8_t> h_rgb;
    std::vector<float> h_tmp;

    uint8_t* d_rgb;
    float* d_input, * d_output;

    size_t workspace_bytes = 0;

    void* d_workspace = nullptr;

    float max_value = 0;
public:
    Convolution(int input_height, int input_width, int input_channels, int filter_dim) 
        : input_height(input_height), input_width(input_width), input_channels(input_channels), input_number(1), filter_dim(filter_dim), filter(filter_dim, input_channels) {
        create_descriptors_and_alacator();
        allocate_memory();
    }

    ~Convolution() {
        // Очистка ресурсов
        CUDNN_CHECK_ERROR(cudnnDestroyTensorDescriptor(input_descriptor));
        CUDNN_CHECK_ERROR(cudnnDestroyConvolutionDescriptor(conv_descriptor));
        CUDNN_CHECK_ERROR(cudnnDestroyTensorDescriptor(output_descriptor));

        CUDA_CHECK_ERROR(cudaFree(d_workspace));
        CUDNN_CHECK_ERROR(cudnnDestroy(cudnn));

        CUDA_CHECK_ERROR(cudaFree(d_rgb));
        CUDA_CHECK_ERROR(cudaFree(d_input));
        CUDA_CHECK_ERROR(cudaFree(d_output));
    }

    void load_image(const uint8_t* h_input) {
        for (int i = 0; i < input_size; ++i) {
            h_tmp[i] = h_input[i];
        }

        // Copy Input to GPU
        CUDA_CHECK_ERROR(cudaMemcpy(d_input, h_tmp.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));

        auto input = d_input;
        unsigned int task_size = input_size;
        
        while (task_size != 1) {
            task_size = launch_max_kernel(256, task_size, input, task_size, d_output);

            input = d_output;
        }

        CUDA_CHECK_ERROR(cudaMemcpy(&max_value, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    }

    void copy_output(std::vector<uint8_t>& buffer) {
        auto input = d_output;
        unsigned int task_size = output_size;

        while (task_size != 1) {
            task_size = launch_max_kernel(256, task_size, input, task_size, d_input);

            input = d_input;
        }

        float tmp_max;
        CUDA_CHECK_ERROR(cudaMemcpy(&tmp_max, d_input, sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK_ERROR(cudaMemcpy(h_tmp.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Resize Output
        if (buffer.size() != output_size)
            buffer.resize(output_size);

        for (int i = 0; i < output_size; ++i) {

            float val = std::min(std::max(h_tmp[i], 0.f), 255.f);


            //float val = (h_tmp[i] * max_value / tmp_max) ;

            buffer[i] = static_cast<uint8_t>(val);
        }
    }

    void add_filter() {
        // filter.generate_identity_filter();
        // filter.generate_blure_filter();
        // filter.generate_gaussian_blure_filter();
        // filter.generate_unsharp_masking_filter();
        filter.generate_ridge_filter();
    }

    void apply() {    
        // Выполнение свёртки
        const float alpha = 1.0f, beta = 0.0f;

        CUDNN_CHECK_ERROR(cudnnConvolutionForward(cudnn,
            &alpha,
            input_descriptor, d_input,
            filter.getDesctiptor(), filter.getFilter(),
            conv_descriptor,
            CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
            d_workspace, workspace_bytes,
            &beta,
            output_descriptor, d_output));    
    }

private:
    void create_descriptors_and_alacator() {
        CUDNN_CHECK_ERROR(cudnnCreate(&cudnn));

        // Создание дескрипторов для изображения и фильтра
        CUDNN_CHECK_ERROR(cudnnCreateTensorDescriptor(&input_descriptor));
        CUDNN_CHECK_ERROR(cudnnSetTensor4dDescriptor(input_descriptor,
            CUDNN_TENSOR_NHWC,   // Формат
            CUDNN_DATA_FLOAT,   // Тип данных
            input_number,       // Batch size
            input_channels,      // Количество каналов
            input_height,        // Высота изображения
            input_width));       // Ширина изображения

        int padding = (filter_dim - 1) / 2;

        // Описание свёртки
        CUDNN_CHECK_ERROR(cudnnCreateConvolutionDescriptor(&conv_descriptor));
        CUDNN_CHECK_ERROR(cudnnSetConvolution2dDescriptor(conv_descriptor,
            padding, padding,  // Паддинг
            1, 1,              // Шаг
            1, 1,              // Режим развертки (дилатации)
            CUDNN_CROSS_CORRELATION,  // Тип свёртки
            CUDNN_DATA_FLOAT));       // Тип данных

        // Определение выходных размеров
        CUDNN_CHECK_ERROR(cudnnGetConvolution2dForwardOutputDim(conv_descriptor,
            input_descriptor,
            filter.getDesctiptor(),
            &output_number,
            &output_channels,
            &output_height,
            &output_width));

        // Создание дескриптора для выходного тензора
        CUDNN_CHECK_ERROR(cudnnCreateTensorDescriptor(&output_descriptor));
        CUDNN_CHECK_ERROR(cudnnSetTensor4dDescriptor(output_descriptor,
            CUDNN_TENSOR_NHWC,
            CUDNN_DATA_FLOAT,
            output_number,
            output_channels,
            output_height,
            output_width));

        // Аллокатор для работы cuDNN
        CUDNN_CHECK_ERROR(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
            input_descriptor,
            filter.getDesctiptor(),
            conv_descriptor,
            output_descriptor,
            CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
            &workspace_bytes));
    }

    void allocate_memory() {
        input_size = input_height * input_width * input_channels;
        output_size = output_height * output_width * output_channels;

        // Host Memory
        h_rgb.resize(std::max(input_size, output_size));
        h_tmp.resize(std::max(input_size, output_size));

        // Device Memory
        CUDA_CHECK_ERROR(cudaMalloc(&d_rgb, std::max(input_size, output_size) * sizeof(uint8_t)));
        CUDA_CHECK_ERROR(cudaMalloc(&d_input, input_size * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMalloc(&d_output, output_size * sizeof(float)));

        // Workspace Memory
        CUDA_CHECK_ERROR(cudaMalloc(&d_workspace, workspace_bytes));
    }

    void calculate_kernel_blocks_threads(int task_size, dim3& output_threads, dim3& output_blocks) {
        int num_threads_per_block = 256;
        int num_blocks = (task_size % num_threads_per_block == 0) ? task_size / num_threads_per_block
            : task_size / num_threads_per_block + 1;

        output_threads = dim3(num_threads_per_block, 1, 1);
        output_blocks = dim3(num_blocks, 1, 1);
    }
};

void applyConvolution(const std::vector<uint8_t>& input, int height, int width, int channels, std::vector<uint8_t>& output, int filter_dim) {
    Convolution convolution(height, width, channels, 3);

    convolution.load_image(input.data());
    convolution.add_filter();

    convolution.apply();

    convolution.copy_output(output);
}
