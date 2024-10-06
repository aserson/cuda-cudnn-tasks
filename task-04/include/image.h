#pragma once

#include "timecounter.cuh"

#include <vector>
#include <string>

class ImageBatch {
    std::vector<uint8_t> _images;
    int _height, _width, _channels, _numbers;
    int _imageSize;

    long _bytes;

    std::string _folderName;

    TimeCounter tc;
    std::vector<std::thread> tasks;
public:
    ImageBatch() : ImageBatch("", 0, 0, 0, 3) {}
    ImageBatch(const std::string& foldername) : ImageBatch(foldername, 0, 0, 0, 3) {}
    ImageBatch(const std::string& foldername, const int numbers);
    ImageBatch(const std::string& foldername, const int numbers, const int height, const int width, const int channels);
    ~ImageBatch();
    
    void set(const int numbers, const int height, const int width, const int channels);
    bool cropAndCopy(int i);
    void write(const std::string& foldername = "");

    bool check(const int numbers, const int height, const int width, const int channels) const {
        return (_numbers == numbers && _channels == channels && _height == height && _width == width);
    }

    int getNumbers() const {
        return _numbers;
    }

    int getHeight() const {
        return _height;
    }

    int getWidth() const {
        return _width;
    }

    int getChannels() const {
        return _channels;
    }

    uint8_t* data() {
        return _images.data();
    }

    const uint8_t* data() const {
        return _images.data();
    }

    int size() {
        return _width * _height * _channels;
    }

    uint8_t operator[](const unsigned int index) const {
        return _images[index];
    }

    uint8_t& operator[](const unsigned int index) {
        return _images[index];
    }
};