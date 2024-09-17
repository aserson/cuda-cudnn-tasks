#pragma once

#include <vector>

#include <cudnn.h>

enum FilterName {
    ZERO_FILTER = 0,
    IDENTITY_FILTER,
    SHARPEN_FILTER,
    BLURE_FILTER,
    GAUSSIAN_BLURE_FILTER,
    UNSHARP_MASKING_FILTER,
    RIDGE_FILTER,
    WEEK_RIDGE_FILTER
};

class Filter {
    int dim = 0, channels = 0;
    int size = 0, bytes = 0;

    std::vector<float> tmp;
    float* filter = nullptr;
    float *grad = nullptr;

public:
    Filter() {};
    ~Filter();

    void set_filter(float* input_filter, int filter_dim, int channels);
    void set_grad();

    bool generate_filter(FilterName name, int generated_dim, int generated_channels);

    float* get_filter() { return filter; }
    const float* get_filter() const { return filter; }

    float* get_gradient() { return grad; }
    const float* get_gradient() const { return grad; }

    int get_size() { return size; }

private:
    void generate_identity_filter();
    void generate_sharpen_filter();
    void generate_blure_filter();
    void generate_gaussian_blure_filter();
    void generate_unsharp_masking_filter();
    void generate_ridge_filter();
    void generate_week_ridge_filter();

};
