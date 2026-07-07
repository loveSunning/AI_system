#include <cute/tensor.hpp>
#include <cute/util/print.hpp>

#include <array>
#include <cstdlib>
#include <iostream>

namespace {

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

  constexpr int kM = 8;
  constexpr int kK = 12;
  std::array<int, kM * kK> global_storage{};
  for (int m = 0; m < kM; ++m) {
    for (int k = 0; k < kK; ++k) {
      global_storage[m * kK + k] = m * 100 + k;
    }
  }

  auto gmem_layout = make_layout(make_shape(Int<8>{}, Int<12>{}), make_stride(Int<12>{}, Int<1>{}));
  auto gA = make_tensor(make_gmem_ptr(global_storage.data()), gmem_layout);

  auto cta_tiler = Shape<_4, _3>{};
  auto cta_coord = make_coord(1, 2);
  auto tAgA = local_tile(gA, cta_tiler, cta_coord);

  auto smem_layout = make_layout(make_shape(Int<4>{}, Int<3>{}), make_stride(Int<1>{}, Int<4>{}));
  std::array<int, decltype(cosize(smem_layout))::value> shared_storage{};
  auto sA = make_tensor(make_smem_ptr(shared_storage.data()), smem_layout);

  for (int i = 0; i < size(sA); ++i) {
    sA(i) = tAgA(i);
  }

  auto rA = make_tensor<int>(Shape<_4, _3>{}, LayoutRight{});
  for (int i = 0; i < size(rA); ++i) {
    rA(i) = sA(i);
  }

  auto thr_layout = make_layout(make_shape(Int<2>{}, Int<3>{}), make_stride(Int<3>{}, Int<1>{}));
  int thr_idx = 4;
  auto tAsA = local_partition(sA, thr_layout, thr_idx);

  std::cout << "CuTe tensor/local_tile/partition demo\n";
  print_object("gA     ", gA);
  print_object("tAgA   ", tAgA);
  print_object("sA     ", sA);
  print_object("rA     ", rA);
  print_object("tAsA   ", tAsA);
  std::cout << "thr_layout: ";
  print(thr_layout);
  std::cout << "\n";

  bool ok = true;
  ok = expect_value("gA", gA, 5, 7, 507) && ok;
  ok = expect_value("tAgA", tAgA, 1, 1, 507) && ok;
  ok = expect_value("sA", sA, 1, 1, 507) && ok;
  ok = expect_value("rA", rA, 1, 1, 507) && ok;

  int partition_value = tAsA(0, 0);
  std::cout << "tAsA(0, 0) = " << partition_value << " expected 507\n";
  ok = (partition_value == 507) && ok;

  if (!ok) {
    std::cerr << "tensor/local_tile/partition check failed\n";
    return EXIT_FAILURE;
  }

  std::cout << "tensor/local_tile/partition check passed\n";
  return EXIT_SUCCESS;
}
