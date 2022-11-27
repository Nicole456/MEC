#include <omp.h>
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

  int IN = atoi(argv[1]);  // 32
  int IH = atoi(argv[2]);  // 224
  int IW = atoi(argv[3]);  // 224
  int IC = atoi(argv[4]);  // 3
  int KH = atoi(argv[5]);  // 3
  int KW = atoi(argv[6]);  // 3
  int KC = atoi(argv[7]);  // 64
  int T = atoi(argv[8]);
  int THREAD_NUM = atoi(argv[9]);

  int OW = IW - KW + 1;
  int OH = IH - KH + 1;

  Tensor<float, 4, RowMajor> input(IN, IH, IW, IC);
  Tensor<float, 4, RowMajor> kernel(KH, KW, IC, KC);
  Tensor<float, 4, RowMajor> output(IN, OH, OW, KC);

  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();

  // for (int h = 0; h < KH; h++) {
  //   for (int w = 0; w < KW; w++) {
  //     for (int c = 0; c < IC; c++) {
  //       for (int n = 0; n < KC; n++) {
  //         kernel(h, w, c, n) = n * (IC * KH * KW) + c * (KH * KW) + h * KW +
  //         w;
  //       }
  //     }
  //   }
  // }

  // for (int n = 0; n < IN; n++) {
  //   for (int h = 0; h < IH; h++) {
  //     for (int w = 0; w < IW; w++) {
  //       for (int c = 0; c < IC; c++) {
  //         input(n, h, w, c) = n * (IC * IH * IW) + c * (IH * IW) + h * IW +
  //         w;
  //       }
  //     }
  //   }
  // }

  int output_size = IN * OH * OW * KC;
  int im2col_size = IN * OW * IH * KW * IC;
  std::cout << "output size " << output_size << std::endl;
  std::cout << "lower size " << im2col_size << std::endl;

  float *im2col = (float *)malloc(sizeof(float) * im2col_size);

  for (int n = 0; n < IN; n++) {
    for (int w = 0; w < OW; w++) {
      int row = 0;
      for (int h = 0; h < IH; h++) {
        for (int k = 0; k < KW; k++) {
          for (int c = 0; c < IC; c++) {
            if (output_size < im2col_size) {
              im2col[row * OW * IN + n * OW + w] = input(n, h, w + k, c);
            } else {
              im2col[row * OW + n * OW * IH * KW * IC + w] =
                  input(n, h, w + k, c);
            }
            row++;
          }
        }
      }
    }
  }

  // cout << "Input I-------------------------" << endl;
  // for (int n = 0; n < IN; n++) {
  //   for (int c = 0; c < IC; c++) {
  //     for (int h = 0; h < IH; h++) {
  //       for (int w = 0; w < IW; w++) {
  //         cout << input(n, h, w, c) << " ";
  //       }
  //       cout << "\n";
  //     }
  //     cout << "\n";
  //   }
  //   cout << "\n";
  // }
  // cout << "Kernel K-------------------------" << endl;
  // for (int n = 0; n < KC; n++) {
  //   for (int c = 0; c < IC; c++) {
  //     for (int h = 0; h < KH; h++) {
  //       for (int w = 0; w < KW; w++) {
  //         cout << kernel(h, w, c, n) << " ";
  //       }
  //       cout << "\n";
  //     }
  //     cout << "\n";
  //   }
  //   cout << "\n";
  // }
  // cout << "im2col L-------------------------" << endl;
  // if (output_size < im2col_size) {
  //   for (int i = 0; i < IH * KW * IC; i++) {
  //     for (int j = 0; j < OW * IN; j++) {
  //       cout << im2col[i * OW * IN + j] << ", ";
  //       if (j == OW * IN - 1) cout << endl;
  //     }
  //   }
  // } else {
  //   for (int i = 0; i < IH * KW * IC * IN; i++) {
  //     for (int j = 0; j < OW; j++) {
  //       cout << im2col[i * OW + j] << ", ";
  //       if (j == OW - 1) cout << endl;
  //     }
  //   }
  // }

  clock_gettime(CLOCK_REALTIME, &start);

  if (OW <= T && output_size < im2col_size) {
    std::cout << "Solution A" << std::endl;

#pragma omp parallel for num_threads(THREAD_NUM)
    for (int h = 0; h < OH; h++) {
      cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                  IN * OW,                           // q row after trans
                  KC,                                // k col
                  KH * KW * IC,                      // q col
                  1,                                 // alpha
                  im2col + h * IN * IC * OW * KW,    // q
                  IN * OW,                           // q col before trans
                  kernel.data(),                     // k
                  KC,                                // k col
                  0,                                 // beta
                  output.data() + h * IN * OW * KC,  // output
                  KC                                 // output col
      );
    }

    // reformat
    std::copy(output.data(), output.data() + IN * OH * OW * KC, im2col);
    for (int n = 0; n < IN; n++) {
      for (int h = 0; h < OH; h++) {
        for (int w = 0; w < OW; w++) {
          for (int c = 0; c < KC; c++) {
            output(n, h, w, c) =
                im2col[h * IN * OW * KC + n * OW * KC + w * KC + c];
          }
        }
      }
    }

  } else {
    std::cout << "Solution B" << std::endl;

#pragma omp parallel for num_threads(THREAD_NUM)
    for (int n = 0; n < IN; n++) {
      for (int h = 0; h < OH; h++) {
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    OW,                                    // q row after trans
                    KC,                                    // k col
                    KH * KW * IC,                          // q col
                    1,                                     // alpha
                    im2col + (n * IH + h) * IC * OW * KW,  // q
                    OW,                                    // q col before trans
                    kernel.data(),                         // k
                    KC,                                    // k col
                    0,                                     // beta
                    output.data() + (n * OH + h) * OW * KC,  // output
                    KC                                       // output col
        );
      }
    }
  }

  // cout << "Matrix O----------------------" << endl;
  // for (int i = 0; i < OH * IN * OW * KC; i++) {
  //   cout << *(output.data()+i) << " ";
  // }

  clock_gettime(CLOCK_REALTIME, &stop);
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
  //         // if (output(b, j, i, od) != expected) {
  //         //   std::cout << "at od=" << od << " b=" << b << " i=" << i
  //         //             << " j=" << j << " " << output(b, j, i, od) << " vs
  //         "
  //         //             << expected << std::endl;
  //         // }
  //         EigenApprox(output(b, j, i, od), expected);
  //       }
  //     }
  //   }
  // }

  free(im2col);
  return 0;
}