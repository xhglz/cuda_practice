cuda_add_executable(exec_hello src/exec_hello.cu)
target_link_libraries(exec_hello ${CUDA_LIBRARIES})

cuda_add_executable(exec_index src/exec_index.cu)
target_link_libraries(exec_index ${CUDA_LIBRARIES})

cuda_add_executable(exec_matmul 
    src/matmul.cu
    src/exec_matmul.cc
)
target_link_libraries(exec_matmul tools ${CUDA_LIBRARIES} )

cuda_add_executable(exec_memory 
    src/matmul.cu
    src/exec_memory.cc
)
target_link_libraries(exec_memory tools ${CUDA_LIBRARIES} )

cuda_add_executable(exec_device_info src/exec_device_info.cc )
target_link_libraries(exec_device_info ${CUDA_LIBRARIES})

cuda_add_executable(exec_stream 
    src/exec_stream.cc
    src/cuda_steam.cu
    src/matmul.cu
)
target_link_libraries(exec_stream tools ${CUDA_LIBRARIES})

cuda_add_executable(exec_rgb_yuv src/exec_rgb_yuv.cu)
target_link_libraries(exec_rgb_yuv ${CUDA_LIBRARIES})
