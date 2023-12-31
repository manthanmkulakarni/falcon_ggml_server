# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# common 

set(TARGET common)

add_library(${TARGET} OBJECT
    common.h
    common.cpp
    )

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PUBLIC .)
target_compile_features(${TARGET} PUBLIC cxx_std_11)
target_link_libraries(${TARGET} PRIVATE llama)


# falcon_common

set(FALCON_TARGET falcon_common)

add_library(${FALCON_TARGET} OBJECT
    falcon_common.h
    falcon_common.cpp
    )

if (BUILD_SHARED_LIBS)
    set_target_properties(${FALCON_TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${FALCON_TARGET} PUBLIC .)
target_compile_features(${FALCON_TARGET} PUBLIC cxx_std_11)
target_link_libraries(${FALCON_TARGET} PRIVATE libfalcon)

# examples

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (EMSCRIPTEN)
else()
    add_subdirectory(main)
    add_subdirectory(falcon) 
    add_subdirectory(falcon_quantize)
    add_subdirectory(quantize)
    add_subdirectory(quantize-stats)
    add_subdirectory(perplexity)
    add_subdirectory(falcon_perplexity)
    add_subdirectory(embedding)
    add_subdirectory(save-load-state)
    add_subdirectory(benchmark)
    add_subdirectory(baby-llama)
    add_subdirectory(train-text-from-scratch)
    add_subdirectory(simple)
    if (LLAMA_METAL)
        add_subdirectory(metal)
    endif()
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(server)
    endif()
endif()
