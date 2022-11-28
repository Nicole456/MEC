# MEC

## Compile
```
git clone https://github.com/leigao97/MEC.git
cd MEC
git submodule update --init --recursive
mkdir build_x86 && mkdir build_arm
sh build.sh  ## change android ndk path at line 6 before running it
```
## Run
```
sh profile1.sh
sh profile2.sh
sh profile3.sh
sh profile4.sh
sh profile5.sh
```

## Reference
* https://arxiv.org/pdf/1706.06873v1.pdf
* https://github.com/BBuf/Memory-efficient-Convolution-for-Deep-Neural-Network
