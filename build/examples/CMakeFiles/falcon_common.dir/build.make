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
include examples/CMakeFiles/falcon_common.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/CMakeFiles/falcon_common.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/falcon_common.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/falcon_common.dir/flags.make

examples/CMakeFiles/falcon_common.dir/falcon_common.cpp.o: examples/CMakeFiles/falcon_common.dir/flags.make
examples/CMakeFiles/falcon_common.dir/falcon_common.cpp.o: /home/ubuntu/poc/ggllm.cpp/examples/falcon_common.cpp
examples/CMakeFiles/falcon_common.dir/falcon_common.cpp.o: examples/CMakeFiles/falcon_common.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/poc/ggllm.cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/falcon_common.dir/falcon_common.cpp.o"
	cd /home/ubuntu/poc/ggllm.cpp/build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/CMakeFiles/falcon_common.dir/falcon_common.cpp.o -MF CMakeFiles/falcon_common.dir/falcon_common.cpp.o.d -o CMakeFiles/falcon_common.dir/falcon_common.cpp.o -c /home/ubuntu/poc/ggllm.cpp/examples/falcon_common.cpp

examples/CMakeFiles/falcon_common.dir/falcon_common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/falcon_common.dir/falcon_common.cpp.i"
	cd /home/ubuntu/poc/ggllm.cpp/build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/poc/ggllm.cpp/examples/falcon_common.cpp > CMakeFiles/falcon_common.dir/falcon_common.cpp.i

examples/CMakeFiles/falcon_common.dir/falcon_common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/falcon_common.dir/falcon_common.cpp.s"
	cd /home/ubuntu/poc/ggllm.cpp/build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/poc/ggllm.cpp/examples/falcon_common.cpp -o CMakeFiles/falcon_common.dir/falcon_common.cpp.s

falcon_common: examples/CMakeFiles/falcon_common.dir/falcon_common.cpp.o
falcon_common: examples/CMakeFiles/falcon_common.dir/build.make
.PHONY : falcon_common

# Rule to build all files generated by this target.
examples/CMakeFiles/falcon_common.dir/build: falcon_common
.PHONY : examples/CMakeFiles/falcon_common.dir/build

examples/CMakeFiles/falcon_common.dir/clean:
	cd /home/ubuntu/poc/ggllm.cpp/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/falcon_common.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/falcon_common.dir/clean

examples/CMakeFiles/falcon_common.dir/depend:
	cd /home/ubuntu/poc/ggllm.cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/poc/ggllm.cpp /home/ubuntu/poc/ggllm.cpp/examples /home/ubuntu/poc/ggllm.cpp/build /home/ubuntu/poc/ggllm.cpp/build/examples /home/ubuntu/poc/ggllm.cpp/build/examples/CMakeFiles/falcon_common.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/falcon_common.dir/depend
