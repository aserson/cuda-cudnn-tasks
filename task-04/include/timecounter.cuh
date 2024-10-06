#pragma once

#include <cuda_runtime.h>

#include <string>

class TimeCounter {
private:
    cudaEvent_t _start, _stop;
    float time;

public:
    TimeCounter();
    TimeCounter(const std::string& startMessage);
    ~TimeCounter();

    void restart(const std::string& startMessage);
    void done(const std::string& stopMessage);

    void start();
    void stop();

    float getTime();
};
