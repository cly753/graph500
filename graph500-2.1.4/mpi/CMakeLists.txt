cmake_minimum_required(VERSION 3.5)
project(mpi)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(/Users/cly/Dropbox/code/isc/graph500/graph500-2.1.4/mpi/headers)

set(SOURCE_FILES
    bfs_custom.c
    bfs_one_sided.c
    bfs_replicated.c
    bfs_replicated_csc.c
    bfs_simple.c
    common.h
    main.c
    mpi_workarounds.h
    oned_csc.c
    oned_csc.h
    oned_csr.c
    oned_csr.h
    onesided.c
    onesided.h
    onesided_emul.c
    redistribute.h
    utils.c
    validate.c)

add_executable(mpi ${SOURCE_FILES})
