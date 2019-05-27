cmake_minimum_required(VERSION 3.12)
project(kerasify)

set(CMAKE_CXX_STANDARD 14)

add_executable(kerasify main.cpp keras_model.cc )
