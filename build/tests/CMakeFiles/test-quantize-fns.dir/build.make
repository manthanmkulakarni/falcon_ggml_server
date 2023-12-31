# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/poc/ggllm.cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/poc/ggllm.cpp/build

# Include any dependencies generated for this target.
include tests/CMakeFiles/test-quantize-fns.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/test-quantize-fns.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/test-quantize-fns.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/test-quantize-fns.dir/flags.make

tests/CMakeFiles/test-quantize-fns.dir/test-quantize-fns.cpp.o: tests/CMakeFiles/test-quantize-fns.dir/flags.make
tests/CMakeFiles/test-quantize-fns.dir/test-quantize-fns.cpp.o: /home/ubuntu/poc/ggllm.cpp/tests/test-quantize-fns.cpp
tests/CMakeFiles/test-quantize-fns.dir/test-quantize-fns.cpp.o: tests/CMakeFiles/test-quantize-fns.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/poc/ggllm.cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/test-quantize-fns.dir/test-quantize-fns.cpp.o"
	cd /home/ubuntu/poc/ggllm.cpp/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/test-quantize-fns.dir/test-quantize-fns.cpp.o -MF CMakeFiles/test-quantize-fns.dir/test-quantize-fns.cpp.o.d -o CMakeFiles/test-quantize-fns.dir/test-quantize-fns.cpp.o -c /home/ubuntu/poc/ggllm.cpp/tests/test-quantize-fns.cpp

tests/CMakeFiles/test-quantize-fns.dir/test-quantize-fns.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test-quantize-fns.dir/test-quantize-fns.cpp.i"
	cd /home/ubuntu/poc/ggllm.cpp/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/poc/ggllm.cpp/tests/test-quantize-fns.cpp > CMakeFiles/test-quantize-fns.dir/test-quantize-fns.cpp.i

tests/CMakeFiles/test-quantize-fns.dir/test-quantize-fns.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test-quantize-fns.dir/test-quantize-fns.cpp.s"
	cd /home/ubuntu/poc/ggllm.cpp/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/poc/ggllm.cpp/tests/test-quantize-fns.cpp -o CMakeFiles/test-quantize-fns.dir/test-quantize-fns.cpp.s

# Object files for target test-quantize-fns
test__quantize__fns_OBJECTS = \
"CMakeFiles/test-quantize-fns.dir/test-quantize-fns.cpp.o"

# External object files for target test-quantize-fns
test__quantize__fns_EXTERNAL_OBJECTS =

bin/test-quantize-fns: tests/CMakeFiles/test-quantize-fns.dir/test-quantize-fns.cpp.o
bin/test-quantize-fns: tests/CMakeFiles/test-quantize-fns.dir/build.make
bin/test-quantize-fns: libllama.a
bin/test-quantize-fns: /usr/local/cuda/lib64/libcudart.so
bin/test-quantize-fns: /usr/local/cuda/lib64/libcublas.so
bin/test-quantize-fns: /usr/local/cuda/lib64/libculibos.a
bin/test-quantize-fns: /usr/local/cuda/lib64/libcublasLt.so
bin/test-quantize-fns: tests/CMakeFiles/test-quantize-fns.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/poc/ggllm.cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/test-quantize-fns"
	cd /home/ubuntu/poc/ggllm.cpp/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-quantize-fns.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/test-quantize-fns.dir/build: bin/test-quantize-fns
.PHONY : tests/CMakeFiles/test-quantize-fns.dir/build

tests/CMakeFiles/test-quantize-fns.dir/clean:
	cd /home/ubuntu/poc/ggllm.cpp/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/test-quantize-fns.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/test-quantize-fns.dir/clean

tests/CMakeFiles/test-quantize-fns.dir/depend:
	cd /home/ubuntu/poc/ggllm.cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/poc/ggllm.cpp /home/ubuntu/poc/ggllm.cpp/tests /home/ubuntu/poc/ggllm.cpp/build /home/ubuntu/poc/ggllm.cpp/build/tests /home/ubuntu/poc/ggllm.cpp/build/tests/CMakeFiles/test-quantize-fns.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/test-quantize-fns.dir/depend

