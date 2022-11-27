# g++ blas_conv.cpp -I./eigen /usr/lib/x86_64-linux-gnu/libcblas.so.3 -o blas_conv
# g++ blas_mec.cpp -I./eigen /usr/lib/x86_64-linux-gnu/libcblas.so.3 -fopenmp -o blas_conv


# ./blas_conv 32 24 24 96 5 5 256
# ./blas_mec 32 24 24 96 5 5 256 100 1
# ./blas_mec 32 24 24 96 5 5 256 100 2
# ./blas_mec 32 24 24 96 5 5 256 100 4
# ./blas_mec 32 24 24 96 5 5 256 100 8
# ./blas_mec 32 24 24 96 5 5 256 100 16
# ./blas_mec 32 24 24 96 5 5 256 100 32
# ./blas_mec 32 24 24 96 5 5 256 100 64

ANDROID_NDK=/home/lei/Tool/android-ndk-r25b
cd build
cmake ..  \
      -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DANDROID_ABI="arm64-v8a" \
      -DANDROID_STL=c++_static \
      -DCMAKE_BUILD_TYPE=Release \
      -DANDROID_NATIVE_API_LEVEL=android-32 \
      -DANDROID_TOOLCHAIN=clang || exit 1;
make -j16 || exit 1;