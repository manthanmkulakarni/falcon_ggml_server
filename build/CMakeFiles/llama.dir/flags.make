# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# compile C with /usr/bin/cc
# compile CUDA with /usr/local/cuda/bin/nvcc
# compile CXX with /usr/bin/c++
C_DEFINES = -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_DMMV_Y=1 -DGGML_PERF=1 -DGGML_USE_CUBLAS -DGGML_USE_K_QUANTS -DK_QUANTS_PER_ITERATION=2

C_INCLUDES = -I/home/ubuntu/poc/ggllm.cpp/. -isystem /usr/local/cuda/include

C_FLAGS = -O3 -DNDEBUG -std=gnu11 -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -mf16c -mfma -mavx -mavx2 -pthread

CUDA_DEFINES = -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_DMMV_Y=1 -DGGML_PERF=1 -DGGML_USE_CUBLAS -DGGML_USE_K_QUANTS -DK_QUANTS_PER_ITERATION=2

CUDA_INCLUDES = --options-file CMakeFiles/llama.dir/includes_CUDA.rsp

CUDA_FLAGS = -O3 -DNDEBUG -std=c++11 -mf16c -mfma -mavx -mavx2 -Xcompiler -pthread

CXX_DEFINES = -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_DMMV_Y=1 -DGGML_PERF=1 -DGGML_USE_CUBLAS -DGGML_USE_K_QUANTS -DK_QUANTS_PER_ITERATION=2

CXX_INCLUDES = -I/home/ubuntu/poc/ggllm.cpp/. -isystem /usr/local/cuda/include

CXX_FLAGS = -O3 -DNDEBUG -std=gnu++11 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-multichar -mf16c -mfma -mavx -mavx2 -pthread
