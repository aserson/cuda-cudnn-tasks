#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image/stb_image.h"
#include "stb_image/stb_image_write.h"

Image::Image() : Image(0, 0, 0) {}

Image::Image(int height, int width, int channels) : _height(height), _width(width), _channels(channels) {
    _vec.resize(height * width * channels);
}

Image::Image(const std::string& filename) {
    load(filename);
}

void Image::load(const std::string& filename) {
    _channels = 3;
    uint8_t* data = stbi_load(filename.c_str(), &_width, &_height, nullptr, _channels);
    assign(data);
    stbi_image_free(data);
}

void Image::write(const std::string& filename) {
    stbi_write_png(filename.c_str(), _width, _height, _channels, data(), _width * _channels);
}

void Image::assign(const uint8_t* data) {
    _vec.assign(data, data + _width * _height * _channels);
}
