#pragma once
#include <c10/metal/common.h>

template <unsigned N = c10::metal::max_ndim>
struct OrgqrParams {
  int32_t num_batch_dims;

  uint32_t m;
  uint32_t m2;
  uint32_t n;
  uint32_t k;

  ::c10::metal::array<uint32_t, N> A_strides;
  ::c10::metal::array<uint32_t, N> tau_strides;
  ::c10::metal::array<uint32_t, N> H_strides;
  ::c10::metal::array<uint32_t, N> H_sizes;
};

struct UnpackPivotsParams {
  uint32_t perm_batch_stride;
  uint32_t pivots_batch_stride;
  uint32_t dim_size;
};

// Maximum matrix size supported by the native Jacobi EVD Metal kernel.
// The kernel stores two N×N float32 matrices (A and V) in threadgroup memory.
// Apple Silicon supports 32KB threadgroup memory: 2 × 64² × 4 = 32768 bytes.
// Larger matrices fall back to CPU.
#define JACOBI_MAX_N 64

struct EighParams {
  uint32_t n;
  uint32_t upper; // 1 = use upper triangle, 0 = use lower triangle
};
