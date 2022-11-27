#include <Eigen/Dense>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

#include "util.cpp"

using namespace Eigen;

int main() {
  struct timespec start, stop;
  double time;

  int IN = 32;   // 2
  int IH = 224;  // 3
  int IW = 224;  // 3
  int IC = 3;    // 2
  int KH = 3;    // 2
  int KW = 3;    // 2
  int KC = 4;    // 2

  // int IN = 2;  // 2
  // int IH = 3;  // 3
  // int IW = 3;  // 3
  // int IC = 2;  // 2
  // int KH = 2;  // 2
  // int KW = 2;  // 2
  // int KC = 2;  // 2

  int OW = IW - KW + 1;
  int OH = IH - KH + 1;

  Tensor<float, 4, RowMajor> input(IN, IH, IW, IC);
  Tensor<float, 4, RowMajor> kernel(KH, KW, IC, KC);

  input = input.constant(11.0f) + input.random();
  kernel = kernel.constant(2.0f) + kernel.random();

  float *lower_buf =
      (float *)malloc(sizeof(float) * IN * OH * OW * KH * KW * IC);
  float *output_buf = (float *)malloc(sizeof(float) * IN * OH * OW * KC);

  TensorMap<Tensor<float, 4, RowMajor>> output(output_buf, IN, OH, OW, KC);

  typedef typename Eigen::internal::traits<
      Eigen::Tensor<float, 4, Eigen::RowMajor>>::Index TensorIndex;
  Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contract_dims;
  contract_dims[0] = Eigen::IndexPair<TensorIndex>(1, 0);

  Eigen::DSizes<TensorIndex, 2> pre_contract_dims;
  pre_contract_dims[1] = KH * KW * IC;
  pre_contract_dims[0] = IN * OH * OW;

  Eigen::DSizes<TensorIndex, 2> kernel_dims;
  kernel_dims[0] = KH * KW * IC;
  kernel_dims[1] = KC;

  Eigen::DSizes<TensorIndex, 4> post_contract_dims;
  post_contract_dims[0] = IN;
  post_contract_dims[1] = OW;
  post_contract_dims[2] = OH;
  post_contract_dims[3] = KC;

  TensorMap<Tensor<float, 2, RowMajor>> lower(lower_buf, IN * OH * OW,
                                              KH * KW * IC);
  lower = input.extract_image_patches(KW, KH, 1, 1, 1, 1, PADDING_VALID)
              .reshape(pre_contract_dims);

  if (clock_gettime(CLOCK_REALTIME, &start) == -1) {
    perror("clock gettime");
  }

  output = lower.contract(kernel.reshape(kernel_dims), contract_dims)
               .reshape(post_contract_dims);

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
