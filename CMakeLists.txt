cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0146 NEW)

project(cuda-cudnn-tasks LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CXX_COMPILER cl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CUDNN_INCLUDE_DIR "C:/Program Files/NVIDIA/CUDNN/v9.4/include/12.6")
set(CUDNN_LIBRARY "C:/Program Files/NVIDIA/CUDNN/v9.4/lib/12.6/x64/cudnn.lib")


include_directories(${CUDNN_INCLUDE_DIR})

add_executable(task-01 task-01/src/main.cpp task-01/src/convolution.cu)
target_include_directories(task-01 PRIVATE ${CMAKE_SOURCE_DIR}/task-01/include)

set_target_properties(task-01 PROPERTIES
    CUDA_ARCHITECTURES "86")

target_link_libraries(task-01 ${CUDNN_LIBRARY})

target_link_options(task-01 PRIVATE /NODEFAULTLIB:LIBCMT)
