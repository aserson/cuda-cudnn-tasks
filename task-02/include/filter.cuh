#pragma once

#include <cudnn.h>

class Filter {
    int dim, channels;
    int size, bytes;

    cudnnFilterDescriptor_t descriptor;

    float* h_filter, * d_filter;

public:
    Filter(int filter_dim, int channels);
    ~Filter();

    void set_filter(float* input_filter);
    void generate_identity_filter();
    void generate_sharpen_filter();
    void generate_blure_filter();
    void generate_gaussian_blure_filter(int size);
    void generate_unsharp_masking_filter();
    void generate_ridge_filter();

    cudnnFilterDescriptor_t& getDesctiptor() { return descriptor; }
    const cudnnFilterDescriptor_t& getDesctiptor() const { return descriptor; }

    float* getFilter() { return d_filter; }
    const float* getFilter() const { return d_filter; }

    int getDim() { return dim; }
    int getChannels() { return channels; }

private:
    void synchronizeWithHost();
    void synchronizeWithDevice();
};
