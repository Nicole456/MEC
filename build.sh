cd build_x86
cmake .. \
      -DBUILD_X86=True || exit 1;
make -j16 || exit 1;

ANDROID_NDK=/home/lei/Tool/android-ndk-r25b
cd build_arm
cmake ..  \
      -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DANDROID_ABI="arm64-v8a" \
      -DANDROID_STL=c++_static \
      -DCMAKE_BUILD_TYPE=Release \
      -DANDROID_NATIVE_API_LEVEL=android-32 \
      -DANDROID_TOOLCHAIN=clang || exit 1;
make -j16 || exit 1;