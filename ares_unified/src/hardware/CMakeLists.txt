# Hardware abstraction library
add_library(ares_hardware STATIC
    cuda_acceleration.cpp
)

target_include_directories(ares_hardware
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

if(ARES_ENABLE_CUDA AND CUDAToolkit_FOUND)
    target_link_libraries(ares_hardware
        PUBLIC CUDA::cublas
        PUBLIC CUDA::curand
        PUBLIC CUDA::cufft
        PUBLIC CUDA::cudnn
        PUBLIC CUDA::cuda_driver
    )
    target_compile_definitions(ares_hardware PUBLIC ARES_ENABLE_CUDA)
endif()