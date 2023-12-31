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
include examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/progress.make

# Include the compile flags for this target's objects.
include examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/flags.make

examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/quantize.cpp.o: examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/flags.make
examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/quantize.cpp.o: /home/ubuntu/poc/ggllm.cpp/examples/falcon_quantize/quantize.cpp
examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/quantize.cpp.o: examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/poc/ggllm.cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/quantize.cpp.o"
	cd /home/ubuntu/poc/ggllm.cpp/build/examples/falcon_quantize && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/quantize.cpp.o -MF CMakeFiles/falcon_quantize.dir/quantize.cpp.o.d -o CMakeFiles/falcon_quantize.dir/quantize.cpp.o -c /home/ubuntu/poc/ggllm.cpp/examples/falcon_quantize/quantize.cpp

examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/quantize.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/falcon_quantize.dir/quantize.cpp.i"
	cd /home/ubuntu/poc/ggllm.cpp/build/examples/falcon_quantize && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/poc/ggllm.cpp/examples/falcon_quantize/quantize.cpp > CMakeFiles/falcon_quantize.dir/quantize.cpp.i

examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/quantize.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/falcon_quantize.dir/quantize.cpp.s"
	cd /home/ubuntu/poc/ggllm.cpp/build/examples/falcon_quantize && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/poc/ggllm.cpp/examples/falcon_quantize/quantize.cpp -o CMakeFiles/falcon_quantize.dir/quantize.cpp.s

# Object files for target falcon_quantize
falcon_quantize_OBJECTS = \
"CMakeFiles/falcon_quantize.dir/quantize.cpp.o"

# External object files for target falcon_quantize
falcon_quantize_EXTERNAL_OBJECTS =

bin/falcon_quantize: examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/quantize.cpp.o
bin/falcon_quantize: examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/build.make
bin/falcon_quantize: liblibfalcon.a
bin/falcon_quantize: libcmpnct_unicode.a
bin/falcon_quantize: /usr/local/cuda/lib64/libcudart.so
bin/falcon_quantize: /usr/local/cuda/lib64/libcublas.so
bin/falcon_quantize: /usr/local/cuda/lib64/libculibos.a
bin/falcon_quantize: /usr/local/cuda/lib64/libcublasLt.so
bin/falcon_quantize: examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/poc/ggllm.cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/falcon_quantize"
	cd /home/ubuntu/poc/ggllm.cpp/build/examples/falcon_quantize && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/falcon_quantize.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/build: bin/falcon_quantize
.PHONY : examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/build

examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/clean:
	cd /home/ubuntu/poc/ggllm.cpp/build/examples/falcon_quantize && $(CMAKE_COMMAND) -P CMakeFiles/falcon_quantize.dir/cmake_clean.cmake
.PHONY : examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/clean

examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/depend:
	cd /home/ubuntu/poc/ggllm.cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/poc/ggllm.cpp /home/ubuntu/poc/ggllm.cpp/examples/falcon_quantize /home/ubuntu/poc/ggllm.cpp/build /home/ubuntu/poc/ggllm.cpp/build/examples/falcon_quantize /home/ubuntu/poc/ggllm.cpp/build/examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/falcon_quantize/CMakeFiles/falcon_quantize.dir/depend

