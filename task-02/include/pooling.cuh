#pragma once

#include <cudnn.h>

class Pooling {
    cudnnPoolingDescriptor_t descriptor;

    int height, width;
    int vertical_stride, horizontal_stride;
public:
    Pooling(int height, int width, int vertical_stride, int horizontal_stride);
    ~Pooling();

    void apply();
private:
};
