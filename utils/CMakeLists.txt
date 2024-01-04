cuda_add_library(tools
    src/timer.cc
)
target_link_libraries(tools ${CUDA_LIBRARIES})
