#include <cute/layout.hpp>
#include <cute/util/print.hpp>

#include <cstdlib>
#include <iostream>

namespace {

template <class Layout>
bool expect_offset(char const* name, Layout const& layout, int i, int j, int expected) {
  int actual = layout(i, j);
  std::cout << name << "(" << i << ", " << j << ") = " << actual << " expected " << expected << "\n";
  return actual == expected;
}

}  // namespace

int main() {
  using namespace cute;

  auto mk_row_major = make_layout(make_shape(Int<4>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));
  auto nk_col_major = make_layout(make_shape(Int<8>{}, Int<4>{}), make_stride(Int<1>{}, Int<8>{}));
  auto smem_bk_stage =
      make_layout(make_shape(Int<16>{}, Int<32>{}, Int<2>{}), make_stride(Int<64>{}, Int<1>{}, Int<32>{}));

  std::cout << "CuTe layout mapping smoke test\n";
  std::cout << "mk_row_major: ";
  print(mk_row_major);
  std::cout << "\n";
  std::cout << "nk_col_major: ";
  print(nk_col_major);
  std::cout << "\n";
  std::cout << "smem_bk_stage: ";
  print(smem_bk_stage);
  std::cout << "\n";

  bool ok = true;
  ok = expect_offset("mk_row_major", mk_row_major, 2, 3, 19) && ok;
  ok = expect_offset("nk_col_major", nk_col_major, 2, 3, 26) && ok;

  int smem_offset = smem_bk_stage(3, 5, 1);
  std::cout << "smem_bk_stage(3, 5, 1) = " << smem_offset << " expected 229\n";
  ok = (smem_offset == 229) && ok;

  if (!ok) {
    std::cerr << "layout mapping check failed\n";
    return EXIT_FAILURE;
  }

  std::cout << "layout mapping check passed\n";
  return EXIT_SUCCESS;
}
