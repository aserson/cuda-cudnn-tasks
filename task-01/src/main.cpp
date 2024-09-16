#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image/stb_image.h"
#include "stb_image/stb_image_write.h"

#include <iostream>
#include <vector>

#include "convolution.h"

int main() {
    int width, height, channels = 3;

    const char* filename = "../task-01/res/cat.png";
    uint8_t* data = stbi_load(filename, &width, &height, nullptr, channels);

    std::vector<uint8_t> input;
    input.assign(data, data + width * height * channels); 

    std::vector<uint8_t> output(width * height * channels, 0);
    applyConvolution(input, height, width, channels, output);

    stbi_write_png("../task-01/res/output.png", width, height, channels, output.data(), width * channels);

    return 0;
}