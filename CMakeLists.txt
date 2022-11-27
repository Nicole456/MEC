cmake_minimum_required(VERSION 3.4)

IF (${BUILD_X86})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
link_libraries(${CMAKE_SOURCE_DIR}/lib/libcblas_x86.so)
ELSE()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-openmp -fopenmp")
link_libraries(${CMAKE_SOURCE_DIR}/lib/libcblas_arm.so)
ENDIF()

include_directories(${CMAKE_SOURCE_DIR}/include/eigen)

add_executable(blas_mec blas_mec.cpp)
add_executable(blas_conv blas_conv.cpp)
