cmake_minimum_required(VERSION 3.12.3)
project(LIF)
set(CMAKE_CXX_STANDARD 14)
find_package(Ceres REQUIRED)

include_directories(${CERES_INCLUDE_DIRS})

include_directories(LIF ../kerasify)
add_executable(LIF main.cpp rmi.cpp rmi.h NeuralNetwork.cpp
        linearReg.cpp linearReg.h multivariateLinearReg.cpp multivariateLinearReg.h Model.cpp Model.h ../kerasify/keras_model.cc ../kerasify/keras_model.h Stage.cpp Stage.h modelsEnum.cpp Test.cpp Test.h)
target_link_libraries(LIF ${CERES_LIBRARIES})

