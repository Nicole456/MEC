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

  int output_size = IN * OH * OW * KC;
  int lower_size = IN * OW * IH * KW * IC;
  float *lower_buf = (float *)malloc(sizeof(float) * lower_size);
  float *output_buf = (float *)malloc(sizeof(float) * output_size);

  std::cout << "output size " << output_size << std::endl;
  std::cout << "lower size " << lower_size << std::endl;

  typedef typename Eigen::internal::traits<
      Eigen::Tensor<float, 4, Eigen::RowMajor>>::Index TensorIndex;
  Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contract_dims;
  contract_dims[0] = Eigen::IndexPair<TensorIndex>(1, 0);

  Eigen::DSizes<TensorIndex, 2> pre_contract_dims;
  pre_contract_dims[0] = IN * OW;
  pre_contract_dims[1] = IH * KW * IC;

  TensorMap<Tensor<float, 2, RowMajor>> lower(lower_buf, IN * OW, IH * KW * IC);

  for (int n = 0; n < IN; n++) {
    for (int w = 0; w < OW; w++) {
      int row = 0;
      for (int h = 0; h < IH; h++) {
        for (int k = 0; k < KW; k++) {
          for (int c = 0; c < IC; c++) {
            if (output_size < lower_size) {
              lower_buf[row * OW * IN + n * OW + w] = input(n, h, w + k, c);
            } else {
              lower_buf[row * OW + n * IH * KW * IC * IN + w] =
                  input(n, h, w + k, c);
            }
            row++;
          }
        }
      }
    }
  }

  Map<Matrix<float, -1, -1, RowMajor>> kernel_mat(kernel.data(), KH * KW * IC,
                                                  KC);
  Map<Matrix<float, -1, -1, RowMajor>> output_mat(output_buf, OH, IN * OW * KC);

  clock_gettime(CLOCK_REALTIME, &start);

  if (output_size < lower_size) {
    std::cout << "Solution A" << std::endl;

    for (int oh = 0; oh < OH; ++oh) {
      Map<Matrix<float, -1, -1, RowMajor>> lower_slice(
          lower_buf + IN * OW * KW * IC * oh, KH * KW * IC, IN * OW);
      Map<Matrix<float, -1, -1, RowMajor>> row(output_mat.row(oh).data(), KC,
                                               IN * OW);
      row.noalias() = (kernel_mat.transpose() * lower_slice);
    }

    std::copy(output_buf, output_buf + IN * OH * OW * KC, lower_buf);

    TensorMap<Tensor<float, 4, RowMajor>> output_ten(lower_buf, OH, KC, IN, OW);
    // std::cout << output_ten << std::endl << std::endl;
    Eigen::array<Eigen::DenseIndex, 4> shuffle = {2, 0, 3, 1};
    TensorMap<Tensor<float, 4, RowMajor>> output(output_buf, IN, OH, OW, KC);
    output = output_ten.shuffle(shuffle);
    // std::cout << output << std::endl;
  } else {
    std::cout << "Solution B" << std::endl;

    for (int in = 0; in < IN; ++in) {
      for (int oh = 0; oh < OH; ++oh) {
        Map<Matrix<float, -1, -1, RowMajor>> lower_slice(
            lower_buf + (in * IH + oh) * IC * OW * KW, KH * KW * IC, OW);
        Map<Matrix<float, -1, -1, RowMajor>> row(output_mat.row(oh).data(), KC,
                                                 IN * OW);
        row.noalias() = (kernel_mat.transpose() * lower_slice);
      }
    }
  }

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
