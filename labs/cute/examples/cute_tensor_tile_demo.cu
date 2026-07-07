#include <cute/tensor.hpp>
#include <cute/util/print.hpp>

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr int kM = 2048;
constexpr int kN = 2048;
constexpr int kK = 2048;

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 32;

constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 16;

constexpr int kBlockCoordM = 3;
constexpr int kBlockCoordN = 5;
constexpr int kBlockCoordK = 7;

int a_value(int m, int k) {
  return m * 100000 + k;
}

int b_value(int n, int k) {
  return n * 100000 + k;
}

int c_value(int m, int n) {
  return m * 100000 + n;
}

template <class Tensor>
void print_object(char const* name, Tensor const& tensor) {
  std::cout << name << ": ";
  cute::print(tensor);
  std::cout << "\n";
}

template <class Tensor>
bool expect_value(char const* name, Tensor const& tensor, int i, int j, int expected) {
  int actual = tensor(i, j);
  std::cout << name << "(" << i << ", " << j << ") = " << actual << " expected " << expected << "\n";
  return actual == expected;
}

}  // namespace

int main() {
  using namespace cute;

  std::vector<int> global_a(kM * kK);
  std::vector<int> global_b(kN * kK);
  std::vector<int> global_c(kM * kN);

  for (int m = 0; m < kM; ++m) {
    for (int k = 0; k < kK; ++k) {
      global_a[m * kK + k] = a_value(m, k);
    }
  }

  for (int n = 0; n < kN; ++n) {
    for (int k = 0; k < kK; ++k) {
      global_b[n * kK + k] = b_value(n, k);
    }
  }

  for (int m = 0; m < kM; ++m) {
    for (int n = 0; n < kN; ++n) {
      global_c[m * kN + n] = c_value(m, n);
    }
  }

  auto gmem_a_layout = make_layout(make_shape(Int<kM>{}, Int<kK>{}), make_stride(Int<kK>{}, Int<1>{}));
  auto gmem_b_layout = make_layout(make_shape(Int<kN>{}, Int<kK>{}), make_stride(Int<kK>{}, Int<1>{}));
  auto gmem_c_layout = make_layout(make_shape(Int<kM>{}, Int<kN>{}), make_stride(Int<kN>{}, Int<1>{}));

  auto gA = make_tensor(make_gmem_ptr(global_a.data()), gmem_a_layout);
  auto gB = make_tensor(make_gmem_ptr(global_b.data()), gmem_b_layout);
  auto gC = make_tensor(make_gmem_ptr(global_c.data()), gmem_c_layout);

  auto cta_tiler = Shape<Int<kBlockM>, Int<kBlockN>, Int<kBlockK>>{};
  auto cta_coord = make_coord(kBlockCoordM, kBlockCoordN, kBlockCoordK);

  auto tAgA = local_tile(gA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  auto tBgB = local_tile(gB, cta_tiler, cta_coord, Step<X, _1, _1>{});
  auto tCgC = local_tile(gC, cta_tiler, cta_coord, Step<_1, _1, X>{});

  auto smem_a_layout =
      make_layout(make_shape(Int<kBlockM>{}, Int<kBlockK>{}), make_stride(Int<1>{}, Int<kBlockM>{}));
  auto smem_b_layout =
      make_layout(make_shape(Int<kBlockN>{}, Int<kBlockK>{}), make_stride(Int<1>{}, Int<kBlockN>{}));

  std::vector<int> shared_a(decltype(cosize(smem_a_layout))::value);
  std::vector<int> shared_b(decltype(cosize(smem_b_layout))::value);
  auto sA = make_tensor(make_smem_ptr(shared_a.data()), smem_a_layout);
  auto sB = make_tensor(make_smem_ptr(shared_b.data()), smem_b_layout);

  for (int i = 0; i < size(sA); ++i) {
    sA(i) = tAgA(i);
  }
  for (int i = 0; i < size(sB); ++i) {
    sB(i) = tBgB(i);
  }

  auto rA = make_tensor<int>(Shape<Int<kMmaM>, Int<kMmaK>>{}, LayoutRight{});
  auto rB = make_tensor<int>(Shape<Int<kMmaN>, Int<kMmaK>>{}, LayoutRight{});
  auto rC = make_tensor<int>(Shape<Int<kMmaM>, Int<kMmaN>>{}, LayoutRight{});

  for (int m = 0; m < kMmaM; ++m) {
    for (int k = 0; k < kMmaK; ++k) {
      rA(m, k) = sA(m, k);
    }
  }
  for (int n = 0; n < kMmaN; ++n) {
    for (int k = 0; k < kMmaK; ++k) {
      rB(n, k) = sB(n, k);
    }
  }
  for (int m = 0; m < kMmaM; ++m) {
    for (int n = 0; n < kMmaN; ++n) {
      rC(m, n) = tCgC(m, n);
    }
  }

  auto thr_layout_a = make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));
  int thr_idx = 45;
  auto tAsA = local_partition(sA, thr_layout_a, thr_idx);

  constexpr int kGlobalM = kBlockCoordM * kBlockM;
  constexpr int kGlobalN = kBlockCoordN * kBlockN;
  constexpr int kGlobalK = kBlockCoordK * kBlockK;

  std::cout << "CuTe GEMM-shaped tensor/local_tile/partition demo\n";
  std::cout << "problem: M=" << kM << " N=" << kN << " K=" << kK << "\n";
  std::cout << "cta tile: BLK_M=" << kBlockM << " BLK_N=" << kBlockN << " BLK_K=" << kBlockK << "\n";
  std::cout << "mma fragments: A=" << kMmaM << "x" << kMmaK << " B=" << kMmaN << "x" << kMmaK
            << " C=" << kMmaM << "x" << kMmaN << "\n";
  std::cout << "cta coord: (" << kBlockCoordM << ", " << kBlockCoordN << ", " << kBlockCoordK
            << ") origin: M=" << kGlobalM << " N=" << kGlobalN << " K=" << kGlobalK << "\n";

  print_object("gA     ", gA);
  print_object("gB     ", gB);
  print_object("gC     ", gC);
  print_object("tAgA   ", tAgA);
  print_object("tBgB   ", tBgB);
  print_object("tCgC   ", tCgC);
  print_object("sA     ", sA);
  print_object("sB     ", sB);
  print_object("rA     ", rA);
  print_object("rB     ", rB);
  print_object("rC     ", rC);
  print_object("tAsA   ", tAsA);
  std::cout << "thr_layout_a: ";
  print(thr_layout_a);
  std::cout << "\n";

  bool ok = true;
  ok = expect_value("gA", gA, kGlobalM + 5, kGlobalK + 7, a_value(kGlobalM + 5, kGlobalK + 7)) && ok;
  ok = expect_value("gB", gB, kGlobalN + 3, kGlobalK + 7, b_value(kGlobalN + 3, kGlobalK + 7)) && ok;
  ok = expect_value("gC", gC, kGlobalM + 5, kGlobalN + 3, c_value(kGlobalM + 5, kGlobalN + 3)) && ok;

  ok = expect_value("tAgA", tAgA, 5, 7, a_value(kGlobalM + 5, kGlobalK + 7)) && ok;
  ok = expect_value("tBgB", tBgB, 3, 7, b_value(kGlobalN + 3, kGlobalK + 7)) && ok;
  ok = expect_value("tCgC", tCgC, 5, 3, c_value(kGlobalM + 5, kGlobalN + 3)) && ok;

  ok = expect_value("sA", sA, 5, 7, a_value(kGlobalM + 5, kGlobalK + 7)) && ok;
  ok = expect_value("sB", sB, 3, 7, b_value(kGlobalN + 3, kGlobalK + 7)) && ok;

  ok = expect_value("rA", rA, 5, 7, a_value(kGlobalM + 5, kGlobalK + 7)) && ok;
  ok = expect_value("rB", rB, 3, 7, b_value(kGlobalN + 3, kGlobalK + 7)) && ok;
  ok = expect_value("rC", rC, 5, 3, c_value(kGlobalM + 5, kGlobalN + 3)) && ok;

  int partition_value = tAsA(0, 0);
  int partition_expected = a_value(kGlobalM + 5, kGlobalK + 5);
  std::cout << "tAsA(0, 0) = " << partition_value << " expected " << partition_expected << "\n";
  ok = (partition_value == partition_expected) && ok;

  if (!ok) {
    std::cerr << "tensor/local_tile/partition check failed\n";
    return EXIT_FAILURE;
  }

  std::cout << "tensor/local_tile/partition check passed\n";
  return EXIT_SUCCESS;
}
