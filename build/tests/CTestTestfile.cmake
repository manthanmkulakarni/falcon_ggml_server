# CMake generated Testfile for 
# Source directory: /home/ubuntu/poc/ggllm.cpp/tests
# Build directory: /home/ubuntu/poc/ggllm.cpp/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test-quantize-fns "/home/ubuntu/poc/ggllm.cpp/build/bin/test-quantize-fns")
set_tests_properties(test-quantize-fns PROPERTIES  _BACKTRACE_TRIPLES "/home/ubuntu/poc/ggllm.cpp/tests/CMakeLists.txt;5;add_test;/home/ubuntu/poc/ggllm.cpp/tests/CMakeLists.txt;9;llama_add_test;/home/ubuntu/poc/ggllm.cpp/tests/CMakeLists.txt;0;")
add_test(test-quantize-perf "/home/ubuntu/poc/ggllm.cpp/build/bin/test-quantize-perf")
set_tests_properties(test-quantize-perf PROPERTIES  _BACKTRACE_TRIPLES "/home/ubuntu/poc/ggllm.cpp/tests/CMakeLists.txt;5;add_test;/home/ubuntu/poc/ggllm.cpp/tests/CMakeLists.txt;10;llama_add_test;/home/ubuntu/poc/ggllm.cpp/tests/CMakeLists.txt;0;")
add_test(test-sampling "/home/ubuntu/poc/ggllm.cpp/build/bin/test-sampling")
set_tests_properties(test-sampling PROPERTIES  _BACKTRACE_TRIPLES "/home/ubuntu/poc/ggllm.cpp/tests/CMakeLists.txt;5;add_test;/home/ubuntu/poc/ggllm.cpp/tests/CMakeLists.txt;11;llama_add_test;/home/ubuntu/poc/ggllm.cpp/tests/CMakeLists.txt;0;")
add_test(test-tokenizer-0 "/home/ubuntu/poc/ggllm.cpp/build/bin/test-tokenizer-0" "/home/ubuntu/poc/ggllm.cpp/tests/../models/ggml-vocab.bin")
set_tests_properties(test-tokenizer-0 PROPERTIES  _BACKTRACE_TRIPLES "/home/ubuntu/poc/ggllm.cpp/tests/CMakeLists.txt;5;add_test;/home/ubuntu/poc/ggllm.cpp/tests/CMakeLists.txt;12;llama_add_test;/home/ubuntu/poc/ggllm.cpp/tests/CMakeLists.txt;0;")
