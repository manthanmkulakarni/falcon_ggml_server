# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# compile C with /usr/bin/cc
# compile CUDA with /usr/local/cuda/bin/nvcc
C_DEFINES = -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_DMMV_Y=1 -DGGML_PERF=1 -DGGML_USE_CUBLAS -DGGML_USE_K_QUANTS -DK_QUANTS_PER_ITERATION=2

C_INCLUDES = -I/usr/local/cuda/include

C_FLAGS = -O3 -DNDEBUG -std=gnu11 -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -mf16c -mfma -mavx -mavx2

CUDA_DEFINES = -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_DMMV_Y=1 -DGGML_PERF=1 -DGGML_USE_CUBLAS -DGGML_USE_K_QUANTS -DK_QUANTS_PER_ITERATION=2

CUDA_INCLUDES = --options-file CMakeFiles/ggml_static.dir/includes_CUDA.rsp

CUDA_FLAGS = -O3 -DNDEBUG -std=c++11 -mf16c -mfma -mavx -mavx2

