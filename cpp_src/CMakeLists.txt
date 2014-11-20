cmake_minimum_required(VERSION 2.8)
file(GLOB MAIN_SRCS
    "*.h"
    "*.cpp"
    "*.hpp"
)
set( CMAKE_CXX_FLAGS "-std=c++11 -O2" )
ADD_EXECUTABLE(datamining ${MAIN_SRCS})
