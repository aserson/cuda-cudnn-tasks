#include "image.h"

#include <iostream>
#include <fstream>
#include <map>
#include <filesystem>
#include <thread>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image/stb_image.h"
#include "stb_image/stb_image_write.h"

ImageBatch::ImageBatch(const std::string& foldername, const int numbers, const int height, const int width, const int channels) {
    _folderName = foldername;
    
    set(numbers, height, width, channels);
}

ImageBatch::ImageBatch(const std::string& foldername, const int numbers)
    : ImageBatch(foldername, numbers, 0, 0, 3) {

    tc.restart("Loading images from disk... ");

    if (cropAndCopy(0)) {
        for (int i = 1; i < _numbers; ++i) {
            cropAndCopy(i);
        }
    }

    tc.done("Done. Time: ");
}

ImageBatch::~ImageBatch() {
    for (auto& task : tasks) {
        if (task.joinable())
            task.join();
    }
}

void ImageBatch::set(const int numbers, const int height, const int width, const int channels) {
    _numbers = numbers;
    _height = height;
    _width = width;
    _channels = channels;

    _imageSize = _height * _width * _channels;
    if (_images.size() < _imageSize * _numbers)
        _images.resize(_imageSize * _numbers);

    _bytes = _imageSize * sizeof(uint8_t);
}

bool ImageBatch::cropAndCopy(int i) {
    uint8_t* data;
    int width, height;

    std::string filename = _folderName + "/" + std::to_string(i) + ".png";

    if (!std::filesystem::exists(filename)) {
        // File not exist
        return false;
    }

    data = stbi_load(filename.c_str(), &width, &height, nullptr, _channels);

    if (i == 0) {
        set(_numbers, height, width, _channels);

        std::memcpy(_images.data() + i * _imageSize, data, _bytes);

        stbi_image_free(data);
        return true;
    }

    if (i >= _numbers) {
        // Too much data
        stbi_image_free(data);
        return false;
    }

    if (width == _width && height == _height) {
        std::memcpy(_images.data() + i * _imageSize, data, _bytes);
        stbi_image_free(data);
        return true;
    } else if (width > _width && height >= _height) {

        int bytes = _width * _channels * sizeof(uint8_t);

        for (int row = 0; row < _height; ++row) {
            auto input = data + row * width * _channels;
            auto output = _images.data() + i * _imageSize + row * _width * _channels;

            std::memcpy(output, input, bytes);
        }

        stbi_image_free(data);
        return true;
    } else {
        // Too small image
        stbi_image_free(data);
        return false;
    }
}

void ImageBatch::write(const std::string& foldername) {
    if (foldername != "")
        _folderName = foldername;

    if (!std::filesystem::exists(_folderName)) {
        std::filesystem::create_directories(_folderName);
    }

    tc.restart("Writing images to disk... ");

    for (int i = 0; i < _numbers; ++i) {
        std::string filename = _folderName + "/" + std::to_string(i) + ".png";
        auto data = _images.data() + _imageSize * i;
        stbi_write_png(filename.c_str(), _width, _height, _channels, data, _width * _channels);
    }

    tc.done("Done. Time: ");
}
