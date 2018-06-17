cmake_minimum_required(VERSION 3.2)

project(face_detect C CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -I")

find_package(OpenMP)
if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP_INCLUDE_DIRS: ${OpenMP_INCLUDE_DIRS}")
    message(STATUS "OpenMP_LIBS: ${OpenMP_LIBS}")
    message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP_C_FLAGS: ${OpenMP_C_FLAGS}")
endif()

add_subdirectory(src)
