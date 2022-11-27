./blas_conv 32 224 224 3 3 3 512;
echo "mec a, 1 th";

./blas_mec 32 224 224 3 3 3 512 100 1;
echo "mec a, 2 th";

./blas_mec 32 224 224 3 3 3 512 100 2;
echo "mec a, 4 th";
./blas_mec 32 224 224 3 3 3 512 100 4;

echo "mec a, 8 th";
./blas_mec 32 224 224 3 3 3 512 100 8;

echo "mec a, 16 th";
./blas_mec 32 224 224 3 3 3 512 100 16;

echo "mec a, 32 th";
./blas_mec 32 224 224 3 3 3 512 100 32;

echo "mec a, 64 th";
./blas_mec 32 224 224 3 3 3 512 100 64;
