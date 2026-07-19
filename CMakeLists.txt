cmake_minimum_required(VERSION 4.2)
project(ml_c C)

set(CMAKE_C_STANDARD 11)

include_directories(.)

add_executable(ml_c
        main.c
        matrix.c
        matrix.h
        model.c
        model.h
        mnist.c
        mnist.h
        types.h)
