# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/timian/Documents/Code/learnedIndexes/li/LIF/ceres-solver

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/timian/Documents/Code/learnedIndexes/li/LIF

# Include any dependencies generated for this target.
include internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/depend.make

# Include the progress variables for this target.
include internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/progress.make

# Include the compile flags for this target's objects.
include internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/flags.make

internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/ba_denseschur_auto_threads_test.cc.o: internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/flags.make
internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/ba_denseschur_auto_threads_test.cc.o: ceres-solver/internal/ceres/generated_bundle_adjustment_tests/ba_denseschur_auto_threads_test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/timian/Documents/Code/learnedIndexes/li/LIF/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/ba_denseschur_auto_threads_test.cc.o"
	cd /home/timian/Documents/Code/learnedIndexes/li/LIF/internal/ceres/generated_bundle_adjustment_tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ba_denseschur_auto_threads_test.dir/ba_denseschur_auto_threads_test.cc.o -c /home/timian/Documents/Code/learnedIndexes/li/LIF/ceres-solver/internal/ceres/generated_bundle_adjustment_tests/ba_denseschur_auto_threads_test.cc

internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/ba_denseschur_auto_threads_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ba_denseschur_auto_threads_test.dir/ba_denseschur_auto_threads_test.cc.i"
	cd /home/timian/Documents/Code/learnedIndexes/li/LIF/internal/ceres/generated_bundle_adjustment_tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/timian/Documents/Code/learnedIndexes/li/LIF/ceres-solver/internal/ceres/generated_bundle_adjustment_tests/ba_denseschur_auto_threads_test.cc > CMakeFiles/ba_denseschur_auto_threads_test.dir/ba_denseschur_auto_threads_test.cc.i

internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/ba_denseschur_auto_threads_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ba_denseschur_auto_threads_test.dir/ba_denseschur_auto_threads_test.cc.s"
	cd /home/timian/Documents/Code/learnedIndexes/li/LIF/internal/ceres/generated_bundle_adjustment_tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/timian/Documents/Code/learnedIndexes/li/LIF/ceres-solver/internal/ceres/generated_bundle_adjustment_tests/ba_denseschur_auto_threads_test.cc -o CMakeFiles/ba_denseschur_auto_threads_test.dir/ba_denseschur_auto_threads_test.cc.s

internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/ba_denseschur_auto_threads_test.cc.o.requires:

.PHONY : internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/ba_denseschur_auto_threads_test.cc.o.requires

internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/ba_denseschur_auto_threads_test.cc.o.provides: internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/ba_denseschur_auto_threads_test.cc.o.requires
	$(MAKE) -f internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/build.make internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/ba_denseschur_auto_threads_test.cc.o.provides.build
.PHONY : internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/ba_denseschur_auto_threads_test.cc.o.provides

internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/ba_denseschur_auto_threads_test.cc.o.provides.build: internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/ba_denseschur_auto_threads_test.cc.o


# Object files for target ba_denseschur_auto_threads_test
ba_denseschur_auto_threads_test_OBJECTS = \
"CMakeFiles/ba_denseschur_auto_threads_test.dir/ba_denseschur_auto_threads_test.cc.o"

# External object files for target ba_denseschur_auto_threads_test
ba_denseschur_auto_threads_test_EXTERNAL_OBJECTS =

bin/ba_denseschur_auto_threads_test: internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/ba_denseschur_auto_threads_test.cc.o
bin/ba_denseschur_auto_threads_test: internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/build.make
bin/ba_denseschur_auto_threads_test: lib/libtest_util.a
bin/ba_denseschur_auto_threads_test: lib/libceres.a
bin/ba_denseschur_auto_threads_test: lib/libgtest.a
bin/ba_denseschur_auto_threads_test: /usr/lib/x86_64-linux-gnu/libspqr.so
bin/ba_denseschur_auto_threads_test: /usr/lib/x86_64-linux-gnu/libcholmod.so
bin/ba_denseschur_auto_threads_test: /usr/lib/x86_64-linux-gnu/libccolamd.so
bin/ba_denseschur_auto_threads_test: /usr/lib/x86_64-linux-gnu/libcamd.so
bin/ba_denseschur_auto_threads_test: /usr/lib/x86_64-linux-gnu/libcolamd.so
bin/ba_denseschur_auto_threads_test: /usr/lib/x86_64-linux-gnu/libamd.so
bin/ba_denseschur_auto_threads_test: /usr/lib/x86_64-linux-gnu/liblapack.so
bin/ba_denseschur_auto_threads_test: /usr/lib/x86_64-linux-gnu/libf77blas.so
bin/ba_denseschur_auto_threads_test: /usr/lib/x86_64-linux-gnu/libatlas.so
bin/ba_denseschur_auto_threads_test: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
bin/ba_denseschur_auto_threads_test: /usr/lib/x86_64-linux-gnu/librt.so
bin/ba_denseschur_auto_threads_test: /usr/lib/x86_64-linux-gnu/libcxsparse.so
bin/ba_denseschur_auto_threads_test: /usr/lib/x86_64-linux-gnu/liblapack.so
bin/ba_denseschur_auto_threads_test: /usr/lib/x86_64-linux-gnu/libf77blas.so
bin/ba_denseschur_auto_threads_test: /usr/lib/x86_64-linux-gnu/libatlas.so
bin/ba_denseschur_auto_threads_test: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
bin/ba_denseschur_auto_threads_test: /usr/lib/x86_64-linux-gnu/librt.so
bin/ba_denseschur_auto_threads_test: /usr/lib/x86_64-linux-gnu/libcxsparse.so
bin/ba_denseschur_auto_threads_test: /usr/local/lib/libgflags_nothreads.a
bin/ba_denseschur_auto_threads_test: /usr/lib/x86_64-linux-gnu/libglog.so
bin/ba_denseschur_auto_threads_test: internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/timian/Documents/Code/learnedIndexes/li/LIF/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../bin/ba_denseschur_auto_threads_test"
	cd /home/timian/Documents/Code/learnedIndexes/li/LIF/internal/ceres/generated_bundle_adjustment_tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ba_denseschur_auto_threads_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/build: bin/ba_denseschur_auto_threads_test

.PHONY : internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/build

internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/requires: internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/ba_denseschur_auto_threads_test.cc.o.requires

.PHONY : internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/requires

internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/clean:
	cd /home/timian/Documents/Code/learnedIndexes/li/LIF/internal/ceres/generated_bundle_adjustment_tests && $(CMAKE_COMMAND) -P CMakeFiles/ba_denseschur_auto_threads_test.dir/cmake_clean.cmake
.PHONY : internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/clean

internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/depend:
	cd /home/timian/Documents/Code/learnedIndexes/li/LIF && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/timian/Documents/Code/learnedIndexes/li/LIF/ceres-solver /home/timian/Documents/Code/learnedIndexes/li/LIF/ceres-solver/internal/ceres/generated_bundle_adjustment_tests /home/timian/Documents/Code/learnedIndexes/li/LIF /home/timian/Documents/Code/learnedIndexes/li/LIF/internal/ceres/generated_bundle_adjustment_tests /home/timian/Documents/Code/learnedIndexes/li/LIF/internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : internal/ceres/generated_bundle_adjustment_tests/CMakeFiles/ba_denseschur_auto_threads_test.dir/depend

