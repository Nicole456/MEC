#include <stdio.h>

#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

#include "include/util.h"

extern "C" {
#include "include/cblas.h"
}

using namespace Eigen;

int main(int argc, char **argv) {
  struct timespec start, stop;
  double time;

  // int IN = 2;  // 2
  // int IH = 3;  // 3
  // int IW = 3;  // 3
  // int IC = 2;  // 2
  // int KH = 2;  // 2
  // int KW = 2;  // 2
  // int KC = 2;  // 2

  int IN = atoi(argv[1]);  // 32
  int IH = atoi(argv[2]);  // 224
  int IW = atoi(argv[3]);  // 224
  int IC = atoi(argv[4]);  // 3
  int KH = atoi(argv[5]);  // 3
  int KW = atoi(argv[6]);  // 3
  int KC = atoi(argv[7]);  // 64

  int OW = IW - KW + 1;
  int OH = IH - KH + 1;

  Tensor<float, 4, RowMajor> input(IN, IH, IW, IC);
  Tensor<float, 4, RowMajor> kernel(KH, KW, IC, KC);
  Tensor<float, 4, RowMajor> output(IN, OH, OW, KC);

  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();

  float *im2col = (float *)malloc(sizeof(float) * IN * OW * OH * KH * KW * IC);

  for (int n = 0; n < IN; n++) {
    for (int h = 0; h < OH; h++) {
      for (int w = 0; w < OW; w++) {
        for (int i = 0; i < KH; i++) {
          for (int j = 0; j < KW; j++) {
            for (int c = 0; c < IC; c++) {
              im2col[n * OH * OW * KH * KW * IC + h * OW * KH * KW * IC +
                     w * KH * KW * IC + i * KW * IC + j * IC + c] =
                  input(n, h + i, w + j, c);
            }
          }
        }
      }
    }
  }

  if (clock_gettime(CLOCK_REALTIME, &start) == -1) {
    perror("clock gettime");
  }

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              IN * OW * OH,   // Q's row
              KC,             // K col
              KH * KW * IC,   // Q's col
              1,              // alpha
              im2col,         // matrix Q
              KH * KW * IC,   // Q's col
              kernel.data(),  // matrix K
              KC,             // K's col
              0,              // beta
              output.data(),  // matrix O
              KC              // O's col
  );

  if (clock_gettime(CLOCK_REALTIME, &stop) == -1) {
    perror("clock gettime");
  }

  time = (stop.tv_sec - start.tv_sec) +
         (double)(stop.tv_nsec - start.tv_nsec) / 1e9;

  std::cout << "Execution time = " << time << std::endl;

  double vm, rss;
  mem_usage(vm, rss);
  std::cout << "Virtual Memory: " << vm << " Resident set size: " << rss
            << std::endl;

  // for (int b = 0; b < IN; ++b) {
  //   for (int od = 0; od < KC; ++od) {
  //     for (int i = 0; i < OH; ++i) {
  //       for (int j = 0; j < OW; ++j) {
  //         float expected = 0.0f;
  //         for (int c = 0; c < KH; ++c) {
  //           for (int r = 0; r < KW; ++r) {
  //             for (int id = 0; id < IC; ++id) {
  //               expected += input(b, c + j, r + i, id) * kernel(c, r, id,
  //               od);
  //             }
  //           }
  //         }
  //         if (output(b, j, i, od) != expected) {
  //           std::cout << "at od=" << od << " b=" << b << " i=" << i
  //                     << " j=" << j << " " << output(b, j, i, od) << " vs "
  //                     << expected << std::endl;
  //         }
  //         EigenApprox(output(b, j, i, od), expected);
  //       }
  //     }
  //   }
  // }
  return 0;
}