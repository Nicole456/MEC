cmake_minimum_required(VERSION 3.4)

set(target blas_mec)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-openmp -fopenmp")

add_executable(${target} blas_mec.cpp)
target_include_directories(${target} PRIVATE /home/lei/Homework/eigen)
target_include_directories(${target} PRIVATE /home/lei/Homework/)
target_link_libraries(${target} /home/lei/Tool/android-ndk-r21e/toolchains/renderscript/prebuilt/linux-x86_64/platform/arm64/libblasV8.so)

# find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#     target_link_libraries(${target} OpenMP::OpenMP_CXX)
# endif()