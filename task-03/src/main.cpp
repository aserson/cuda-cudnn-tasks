#include <iostream>
#include <vector>

#include "image.h"
#include "convolution.cuh"

bool applyConvolution(const Image& input, FilterName name, Image& output) {
    Convolution convolution;

    convolution.set_data_desc(input.getHeight(), input.getWidth(), input.getChannels());
    convolution.set_filter_desc(3, input.getChannels());
    convolution.set_convolution();

    convolution.set_input(input);
    convolution.set_filter(RIDGE_FILTER);

    if (convolution.apply_forward()) {
        return convolution.get_output(output);
    }

    return false;
}

bool gradientDescent(const Image& expected, const Image& input, Image& output) {
    Convolution convolution;

    convolution.set_data_desc(input.getHeight(), input.getWidth(), input.getChannels());
    convolution.set_filter_desc(3, input.getChannels());

    convolution.set_input(input);
    convolution.set_expected(expected);
    convolution.set_filter(ZERO_FILTER);

    convolution.set_convolution();
    convolution.set_convolution_bwd();

    if (convolution.gradient_descent()) {
        return convolution.get_output(output);
    }

    return true;
}

int main() {
    Image input("../task-03/res/input.png");
    Image expected("../task-03/res/expected.png");

    Image output;
    gradientDescent(expected, input, output);
    output.write("../task-03/res/output.png");

    return 0;
}