#include <iostream>
#include <vector>

#include "image.h"
#include "convolution.cuh"

bool applyConvolution(const ImageBatch& input, FilterName name, ImageBatch& output) {
    Convolution convolution;

    convolution.set_data_desc(input);
    convolution.set_filter_desc(3, input.getChannels());
    convolution.set_convolution();

    convolution.set_input(input);
    convolution.set_filter(name);

    if (convolution.apply_forward()) {
        return convolution.get_output(output);
    }

    return false;
}

bool gradientDescent(const ImageBatch& expected, const ImageBatch& input, ImageBatch& output) {
    Convolution convolution;

    convolution.set_data_desc(input);
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

void runForward() {
    ImageBatch input("../task-04/res/cats", 1000);
    ImageBatch output("../task-04/res/expected");

    applyConvolution(input, RIDGE_FILTER, output);

    output.write();
}

void runBackward() {
    ImageBatch input("../task-04/res/cats", 1000);
    ImageBatch expected("../task-04/res/expected", 1000);
    ImageBatch output("../task-04/res/outputs");

    gradientDescent(expected, input, output);

    output.write();
}


int main() {
    runForward();

    return 0;
}