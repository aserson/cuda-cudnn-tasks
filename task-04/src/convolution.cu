#include <iostream>
#include <algorithm>
#include <cudnn.h>

#include "convolution.cuh"
#include "launcher.cuh"
#include "kernels.cuh"

Convolution::Convolution(cudnnDataType_t type) : type(type), stop_flag(false) {
    CUDNN_CHECK_ERROR(cudnnCreate(&cudnn));

    // Создание дескрипторов памяти
    CUDNN_CHECK_ERROR(cudnnCreateTensorDescriptor(&data_descriptor));

    // Создание дескриптора под свёртку
    CUDNN_CHECK_ERROR(cudnnCreateConvolutionDescriptor(&conv_descriptor));
    CUDNN_CHECK_ERROR(cudnnCreateFilterDescriptor(&filter_descriptor));
}

Convolution::~Convolution() {
    // Очистка ресурсов
    CUDNN_CHECK_ERROR(cudnnDestroy(cudnn));

    CUDNN_CHECK_ERROR(cudnnDestroyTensorDescriptor(data_descriptor));

    CUDNN_CHECK_ERROR(cudnnDestroyConvolutionDescriptor(conv_descriptor));
    CUDNN_CHECK_ERROR(cudnnDestroyFilterDescriptor(filter_descriptor));

    if (f_workspace != nullptr) CUDA_CHECK_ERROR(cudaFree(f_workspace));
    if (b_workspace != nullptr) CUDA_CHECK_ERROR(cudaFree(b_workspace));

    if (input != nullptr) CUDA_CHECK_ERROR(cudaFree(input));
    if (output != nullptr) CUDA_CHECK_ERROR(cudaFree(output));
    if (expected != nullptr) CUDA_CHECK_ERROR(cudaFree(expected));
    if (output_grad != nullptr) CUDA_CHECK_ERROR(cudaFree(output_grad));
    if (d_tmp != nullptr) CUDA_CHECK_ERROR(cudaFree(d_tmp));
}

bool Convolution::set_data_desc(const ImageBatch& input) {
    if (channels != 0 && channels != input.getChannels()) {
        std::cout << "Data channels not equal filter channels" << std::endl;
        return false;
    }

    if (channels == 0) channels = input.getChannels();
    height = input.getHeight();
    width = input.getWidth();
    numbers = input.getNumbers();

    pic_size = height * width * channels;

    size = pic_size * numbers;
    bytes = size * sizeof(TData);

    CUDNN_CHECK_ERROR(cudnnSetTensor4dDescriptor(data_descriptor,
        CUDNN_TENSOR_NCHW, type,
        numbers, channels, height, width));

    return true;
}

bool Convolution::set_filter_desc(int filter_dim, int filter_channels) {
    if (channels != 0 && channels != filter_channels) {
        std::cout << "Filter channels not equal data channels" << std::endl;
        return false;
    }

    if (channels == 0) channels = filter_channels;
    dim = filter_dim;

    CUDNN_CHECK_ERROR(cudnnSetFilter4dDescriptor(filter_descriptor,
        type, CUDNN_TENSOR_NCHW,
        channels, channels, dim, dim));

    return true;
}

bool Convolution::set_convolution() {
    int padding = (dim - 1) / 2;
    CUDNN_CHECK_ERROR(cudnnSetConvolution2dDescriptor(conv_descriptor,
        padding, padding, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, type));

    int output_numbers, output_channels, output_height, output_width;

    CUDNN_CHECK_ERROR(cudnnGetConvolution2dForwardOutputDim(conv_descriptor,
        data_descriptor, filter_descriptor,
        &output_numbers, &output_channels, &output_height, &output_width));

    if (output_numbers != numbers || output_channels != channels || output_height != height || output_width != width) {
        std::cout << "Output parameters not equal data parameters" << std::endl;
        return false;
    }

    // Creation best algoritm
    int alfoSize = 1;
    CUDNN_CHECK_ERROR(cudnnFindConvolutionForwardAlgorithm(cudnn,
        data_descriptor, filter_descriptor,
        conv_descriptor, data_descriptor,
        alfoSize, &alfoSize, &algo_pref));

    // Allocation Workspace
    CUDNN_CHECK_ERROR(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
        data_descriptor, filter_descriptor, conv_descriptor, data_descriptor, 
        algo_pref.algo, &f_workspace_bytes));

    CUDA_CHECK_ERROR(cudaMalloc(&f_workspace, f_workspace_bytes));

    set_output();

    return true;
}

bool Convolution::set_convolution_bwd() {
    // Creation best algoritm
    int alfoSize = 1;
    CUDNN_CHECK_ERROR(cudnnFindConvolutionBackwardFilterAlgorithm(cudnn,
        data_descriptor, data_descriptor, conv_descriptor, filter_descriptor,
        alfoSize, &alfoSize, &algo_pref_bwd));

    // Allocation Workspace
    CUDNN_CHECK_ERROR(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn,
        data_descriptor, data_descriptor, conv_descriptor, filter_descriptor,
        algo_pref_bwd.algo, &b_workspace_bytes));

    CUDA_CHECK_ERROR(cudaMalloc(&b_workspace, b_workspace_bytes));

    set_output_grad();
    filter.set_grad();
    CUDA_CHECK_ERROR(cudaMalloc(&d_tmp, bytes));

    return true;
}

bool Convolution::set_input(const ImageBatch& h_input) {
    if (!h_input.check(numbers, height, width, channels)) {
        std::cout << "Input parameters not equal data parameters" << std::endl;
        return false;
    }

    auto tmp_data = fromNHWCtoNCHW(h_input);

    CUDA_CHECK_ERROR(cudaMalloc(&input, bytes));
    CUDA_CHECK_ERROR(cudaMemcpy(input, tmp_data, bytes, cudaMemcpyHostToDevice));

    return true;
}

bool Convolution::set_expected(const ImageBatch& h_expected) {
    if (!h_expected.check(numbers, height, width, channels)) {
        std::cout << "Expected parameters not equal data parameters" << std::endl;
        return false;
    }

    auto tmp_data = fromNHWCtoNCHW(h_expected);

    CUDA_CHECK_ERROR(cudaMalloc(&expected, bytes));
    CUDA_CHECK_ERROR(cudaMemcpy(expected, tmp_data, bytes, cudaMemcpyHostToDevice));

    return true;
}

bool Convolution::set_output() {
    CUDA_CHECK_ERROR(cudaMalloc(&output, bytes));

    return true;
}

bool Convolution::set_output_grad() {
    CUDA_CHECK_ERROR(cudaMalloc(&output_grad, bytes));

    return true;
}

bool Convolution::set_filter(FilterName name) {
    return filter.generate_filter(name, dim, channels);
}

bool Convolution::apply_forward() {
    CUDNN_CHECK_ERROR(cudnnConvolutionForward(cudnn,
        &alpha, data_descriptor, input,
        filter_descriptor, filter.get_filter(),
        conv_descriptor, algo_pref.algo,
        f_workspace, f_workspace_bytes, &beta,
        data_descriptor, output));

    return true;
}

bool Convolution::apply_backward() {
    CUDNN_CHECK_ERROR(cudnnConvolutionBackwardFilter(cudnn,
        &alpha, data_descriptor, input,
        data_descriptor, output_grad,
        conv_descriptor, algo_pref_bwd.algo,
        b_workspace, b_workspace_bytes, &beta,
        filter_descriptor, filter.get_gradient()));

     return true;
}

bool Convolution::gradient_descent(int max_cycles) {
    apply_forward();

    TData loss = find_loss_gradient();
    TData rate = 1e-8;

    std::thread input_thread(&Convolution::listen_for_commands, this);

    while (loss > epsilon && count < max_cycles) {
        apply_backward();

        rate = find_learning_rate();
        apply_filter_gradient(rate);

        apply_forward();

        loss = find_loss_gradient();

        if (count % 100 == 0) {
            std::cout << "Cycle: " << count << " Loss: " << loss;
            std::cout << " Rate: " << rate << std::endl;
        }

        count++;

        if (stop_flag) {
            std::cout << "Cycle: " << count << " Loss: " << loss;
            std::cout << " Rate: " << rate << std::endl;
            std::cout << "Stopping the process..." << std::endl;
            break;
        }
    }

    tmp.resize(filter.get_size());
    cudaMemcpy(tmp.data(), filter.get_filter(), 9 * sizeof(TData), cudaMemcpyDeviceToHost);
    std::cout << "Filter : " << std::endl;
    std::cout << tmp[0] << " " << tmp[1] << " " << tmp[2] << " " << std::endl;
    std::cout << tmp[3] << " " << tmp[4] << " " << tmp[5] << " " << std::endl;
    std::cout << tmp[6] << " " << tmp[7] << " " << tmp[8] << " " << std::endl;
    
    input_thread.join();

    return true;
}

bool Convolution::get_output(ImageBatch& h_output) {
    h_output.set(numbers, height, width, channels);

    tmp.resize(size);
    CUDA_CHECK_ERROR(cudaMemcpy(tmp.data(), output, bytes, cudaMemcpyDeviceToHost));

    for (int n = 0; n < numbers; ++n) {
        for (int ch = 0; ch < channels; ++ch) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int input_idx = n * pic_size + ch * width * height + y * width + x;
                    int output_idx = n * pic_size + (y * width + x) * channels + ch;

                    TData value = tmp[input_idx] * 255.f;
                    value = std::min(std::max(value, 0.f), 255.f);
                    h_output[output_idx] = static_cast<uint8_t>(value);
                }
            }
        }
    }

    return true;
}

TData* Convolution::fromNHWCtoNCHW(const ImageBatch& input) {
    tmp.resize(size);

    for (int n = 0; n < numbers; ++n) {
        for (int ch = 0; ch < channels; ++ch) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int input_idx = n * pic_size + (y * width + x) * channels + ch;
                    int output_idx = n * pic_size + ch * width * height + y * width + x;

                    uint8_t value = input[input_idx];
                    tmp[output_idx] = static_cast<TData>(value) / 255.f;
                }
            }
        }
    }

    return tmp.data();
}

TData Convolution::find_loss_gradient() {
    launch_kernel(256, size, find_gradient, size, output, expected, output_grad);

    auto tmp_sum = output_grad;
    unsigned int task_size = size;

    while (task_size > 1) {
        task_size = launch_shared_kernel(256, task_size, find_sum, tmp_sum, task_size, d_tmp);
        tmp_sum = d_tmp;
    }

    TData h_tmp_sum;
    CUDA_CHECK_ERROR(cudaMemcpy(&h_tmp_sum, d_tmp, sizeof(TData), cudaMemcpyDeviceToHost));

    TData ans = h_tmp_sum / size;

    return ans;
}

TData Convolution::find_learning_rate() {
    auto tmp_sum = filter.get_gradient();
    unsigned int task_size = filter.get_size();;

    while (task_size > 1) {
        task_size = launch_shared_kernel(256, task_size, find_max, tmp_sum, task_size, d_tmp);
        tmp_sum = d_tmp;
    }

    TData h_tmp_sum;
    CUDA_CHECK_ERROR(cudaMemcpy(&h_tmp_sum, d_tmp, sizeof(TData), cudaMemcpyDeviceToHost));

    return rate_coef / h_tmp_sum;
}

void Convolution::apply_filter_gradient(TData learning_rate) {
    launch_kernel(256, filter.get_size(), apply_gradient, 
        filter.get_size(), filter.get_filter(), filter.get_gradient(), learning_rate);
}

void Convolution::listen_for_commands() {
    std::string command;
    while (true) {
        std::cin >> command;
        if (command == "exit") {
            stop_flag = true;
            break;
        }
    }
}