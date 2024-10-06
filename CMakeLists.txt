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
include_directories(${CUDNN_INCLUDE_DIR})

set(CUDNN_LIBRARY "C:/Program Files/NVIDIA/CUDNN/v9.4/lib/12.6/x64/cudnn.lib")
link_libraries(${CUDNN_LIBRARY})

foreach(i RANGE 1 4)
    if(${i} LESS 10)
        set(idx "0${i}")
    else()
        set(idx "${i}")
    endif()

    set(TASK_NAME "task-${idx}")
    
    file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/task-${idx}/src/*.cpp" "${CMAKE_SOURCE_DIR}/task-${idx}/src/*.cu")
    file(GLOB HEADERS "${CMAKE_SOURCE_DIR}/task-${idx}/include/*.h" "${CMAKE_SOURCE_DIR}/task-${idx}/include/*.cuh")

    source_group("Source Files" FILES ${SOURCES})
    source_group("Header Files" FILES ${HEADERS})
    add_executable(${TASK_NAME} ${SOURCES} ${HEADERS})

    target_include_directories(${TASK_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/task-${idx}/include)
    set_target_properties(${TASK_NAME} PROPERTIES CUDA_ARCHITECTURES "86")

    target_link_options(${TASK_NAME} PRIVATE /NODEFAULTLIB:LIBCMT)
endforeach()

