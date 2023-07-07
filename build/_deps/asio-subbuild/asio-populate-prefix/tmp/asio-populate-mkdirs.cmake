# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/ubuntu/poc/ggllm.cpp/build/asio/source"
  "/home/ubuntu/poc/ggllm.cpp/build/asio/build"
  "/home/ubuntu/poc/ggllm.cpp/build/asio"
  "/home/ubuntu/poc/ggllm.cpp/build/_deps/asio-subbuild/asio-populate-prefix/tmp"
  "/home/ubuntu/poc/ggllm.cpp/build/_deps/asio-subbuild/asio-populate-prefix/src/asio-populate-stamp"
  "/home/ubuntu/poc/ggllm.cpp/build/_deps/asio-subbuild/asio-populate-prefix/src"
  "/home/ubuntu/poc/ggllm.cpp/build/_deps/asio-subbuild/asio-populate-prefix/src/asio-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ubuntu/poc/ggllm.cpp/build/_deps/asio-subbuild/asio-populate-prefix/src/asio-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ubuntu/poc/ggllm.cpp/build/_deps/asio-subbuild/asio-populate-prefix/src/asio-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
