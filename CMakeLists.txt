cmake_minimum_required(VERSION 3.8)
project(cuda_practice)

find_package(CUDA REQUIRED)

message(STATUS "cuda version: " ${CUDA_VERSION_STRING})

include_directories(
    ${CUDA_INCLUDE_DIRS}
    utils/src
    basic/src
)

add_subdirectory(utils)
add_subdirectory(basic)