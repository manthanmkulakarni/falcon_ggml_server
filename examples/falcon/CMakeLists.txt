set(TARGET falcon_main)
set(ASIO_INCLUDE_DIR "../../asio/asio/include")
add_executable(${TARGET} falcon_main.cpp)

include(FetchContent)

# Set the path where asio will be downloaded and installed
set(ASIO_INSTALL_DIR ${CMAKE_BINARY_DIR}/asio)

# Download and install asio
FetchContent_Declare(
    asio
    GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
    GIT_TAG        master
    SOURCE_DIR     ${ASIO_INSTALL_DIR}/source
    BINARY_DIR     ${ASIO_INSTALL_DIR}/build
    INSTALL_DIR    ${ASIO_INSTALL_DIR}
)
FetchContent_MakeAvailable(asio)

# Add the asio include directory
include_directories(${ASIO_INSTALL_DIR}/source/asio/include)

target_link_libraries(${TARGET} PRIVATE falcon_common libfalcon ${CMAKE_THREAD_LIBS_INIT} )
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()
