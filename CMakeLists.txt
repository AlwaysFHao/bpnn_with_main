cmake_minimum_required(VERSION 3.20)
project(bpnn_with_main C)

set(CMAKE_C_STANDARD 99)

LINK_LIBRARIES(m)
add_executable(bpnn_with_main main.c)
