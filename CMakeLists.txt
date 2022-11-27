cmake_minimum_required(VERSION 3.4)

set(target blas_mec)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-openmp -fopenmp")

add_executable(${target} blas_mec.cpp)
target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/include/eigen)
target_link_libraries(${target} ${CMAKE_SOURCE_DIR}/lib/libcblas_arm.so)

# find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#     target_link_libraries(${target} OpenMP::OpenMP_CXX)
# endif()