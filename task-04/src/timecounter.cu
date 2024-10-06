#include "timecounter.cuh"

#include <iostream>

TimeCounter::TimeCounter() {
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);

    start();
}

TimeCounter::TimeCounter(const std::string& startMessage)
    : TimeCounter::TimeCounter() {
    std::cout << startMessage;
    start();
}

TimeCounter::~TimeCounter() {
    cudaEventDestroy(_stop);
    cudaEventDestroy(_start);
}

void TimeCounter::restart(const std::string& startMessage) {
    std::cout << startMessage;
    start();
}

void TimeCounter::done(const std::string& stopMessage) {
    stop();
    std::cout << stopMessage << getTime() << std::endl;
}

void TimeCounter::start() {
    cudaEventRecord(_start, 0);
}

void TimeCounter::stop() {
    cudaEventRecord(_stop, 0);
    cudaEventSynchronize(_stop);

    cudaEventElapsedTime(&time, _start, _stop);
}

float TimeCounter::getTime() {
    return time / 1000;
}
