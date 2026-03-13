# Minimum version of CMake required
cmake_minimum_required(VERSION 3.10)

# Project name
project(NeuralNetworkFromScratch VERSION 1.0)

# Require C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Tell the compiler where to find the .h files
include_directories(${CMAKE_SOURCE_DIR}/include)

# Tell CMake to gather all .cpp files in the src directory
file(GLOB SOURCES "src/*.cpp")

# Create the final executable and link all the source files together
add_executable(network ${SOURCES})