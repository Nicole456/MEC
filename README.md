# MEC

## Compile
```
git clone https://github.com/leigao97/MEC.git
git submodule update --init --recursive
mkdir build_x86 && mkdir build_arm
sh build.sh  ## change android ndk path at line 6 before running it
```
## Usage
```
./blas_conv 32 24 24 96 5 5 256
./blas_mec 32 24 24 96 5 5 256 100 1
./blas_mec 32 24 24 96 5 5 256 100 2
./blas_mec 32 24 24 96 5 5 256 100 4
./blas_mec 32 24 24 96 5 5 256 100 8
./blas_mec 32 24 24 96 5 5 256 100 16
./blas_mec 32 24 24 96 5 5 256 100 32
./blas_mec 32 24 24 96 5 5 256 100 64
```

## Reference
* https://arxiv.org/pdf/1706.06873v1.pdf
* https://github.com/BBuf/Memory-efficient-Convolution-for-Deep-Neural-Network
