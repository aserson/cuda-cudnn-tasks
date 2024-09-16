#pragma once

void applyConvolution(const std::vector<uint8_t>& input, int height, int width, int channels, std::vector<uint8_t>& output, int filter_dim = 3);