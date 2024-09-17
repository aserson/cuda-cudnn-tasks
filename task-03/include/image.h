#pragma once

#include <vector>
#include <string>

class Image {
    std::vector<uint8_t> _vec;
    int _height, _width, _channels;

public:
    Image();
    Image(int height, int width, int channels);

    Image(const std::string& filename);

    void load(const std::string& filename);

    void write(const std::string& filename);

    void assign(const uint8_t* data);

    bool check(const int height, const int width, const int channels) const {
        return (_channels == channels && _height == height && _width == width);
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
        return _vec.data();
    }

    const uint8_t* data() const {
        return _vec.data();
    }

    int size() {
        return _width * _height * _channels;
    }

    uint8_t operator[](const unsigned int index) const {
        return _vec[index];
    }

    uint8_t& operator[](const unsigned int index) {
        return _vec[index];
    }
};
