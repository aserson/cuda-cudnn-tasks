#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image/stb_image.h"
#include "stb_image/stb_image_write.h"

#include <iostream>
#include <vector>

#include "convolution.cuh"

bool applyConvolution(const std::vector<uint8_t>& input, int height, int width, int channels,
    std::vector<uint8_t>& output, int& output_height, int& output_width, int& output_channels) {
    Convolution convolution(channels);
    convolution.set_input(input.data(), height, width, channels);
    convolution.set_filter(3, channels);

    convolution.set_convolution();
    convolution.apply_convolution();

    output.resize(convolution.get_convolution_output_size());
    if (convolution.get_convolution_output(output.data(), output.size())) {
        output_height = convolution.get_convolution_output_height();
        output_width = convolution.get_convolution_output_width();
        output_channels = convolution.get_convolution_output_channels();

        return true;
    }

    return false;
}

bool applyConvolutionWithPooling(const std::vector<uint8_t>& input, int height, int width, int channels, 
    std::vector<uint8_t>& output, int& output_height, int& output_width, int& output_channels) {
    Convolution convolution(channels);
    convolution.set_input(input.data(), height, width, channels);
    convolution.set_filter(3, channels);

    convolution.set_convolution();
    convolution.apply_convolution();

    convolution.set_pooling(2, 2, 2, 2);
    convolution.apply_pooling();

    output.resize(convolution.get_pooling_output_size());
    if (convolution.get_pooling_output(output.data(), output.size())) {
        output_height = convolution.get_pooling_output_height();
        output_width = convolution.get_pooling_output_width();
        output_channels = convolution.get_pooling_output_channels();

        return true;
    }

    return false;
}

int main() {
    int width, height, channels = 3;

    const char* filename = "../task-02/res/cat.png";
    uint8_t* data = stbi_load(filename, &width, &height, nullptr, channels);

    std::vector<uint8_t> input;
    input.assign(data, data + width * height * channels); 

    std::vector<uint8_t> output;
    int output_height, output_width, output_channels;

    if (applyConvolution(input, height, width, channels, output, output_height, output_width, output_channels))
        stbi_write_png("../task-02/res/c_output.png", output_width, output_height, output_channels, output.data(), output_width * output_channels);
    else
        std::cout << "The task failed" << std::endl;

    if (applyConvolutionWithPooling(input, height, width, channels, output, output_height, output_width, output_channels))
        stbi_write_png("../task-02/res/cp_output.png", output_width, output_height, output_channels, output.data(), output_width * output_channels);
    else
        std::cout << "The task failed" << std::endl;

    return 0;
}