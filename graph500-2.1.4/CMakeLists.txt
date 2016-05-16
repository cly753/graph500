cmake_minimum_required(VERSION 3.5)
project(graph500_2_1_4)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    compat.h
    graph500.c
    graph500.h
    kronecker.c
    kronecker.h
    make-edgelist.c
    options.c
    options.h
    prng.c
    prng.h
    rmat.c
    rmat.h
    timer.c
    timer.h
    verify.c
    verify.h
    xalloc.c
    xalloc.h)

add_executable(graph500_2_1_4 ${SOURCE_FILES})