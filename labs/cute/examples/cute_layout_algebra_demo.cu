#include <cute/tensor.hpp>
#include <cute/util/print.hpp>

#include <cstdlib>
#include <iostream>

namespace {

template <class Object>
void print_object(char const* name, Object const& object) {
  std::cout << name << ": ";
  cute::print(object);
  std::cout << "\n";
}

template <class Value>
bool expect_index(char const* expr, Value const& value, int expected) {
  int actual = value;
  std::cout << expr << " = " << actual << " expected " << expected << "\n";
  return actual == expected;
}

bool expect_true(char const* expr, bool value) {
  std::cout << expr << " = " << (value ? "true" : "false") << " expected true\n";
  return value;
}

template <class LayoutA, class LayoutB>
bool expect_same_1d_mapping(char const* name, LayoutA const& a, LayoutB const& b) {
  bool ok = true;
  for (int i = 0; i < size(a); ++i) {
    int actual = a(i);
    int expected = b(i);
    if (actual != expected) {
      std::cout << name << "(" << i << ") = " << actual << " expected " << expected << "\n";
      ok = false;
    }
  }
  std::cout << name << " same 1D mapping = " << (ok ? "true" : "false") << " expected true\n";
  return ok;
}

template <class LayoutA, class LayoutB, class LayoutR>
bool expect_composition_property(char const* name, LayoutA const& a, LayoutB const& b, LayoutR const& r) {
  bool ok = true;
  for (int i = 0; i < size(b); ++i) {
    int actual = r(i);
    int expected = a(b(i));
    if (actual != expected) {
      std::cout << name << "(" << i << ") = " << actual << " expected " << expected << "\n";
      ok = false;
    }
  }
  std::cout << name << ": R(i) == A(B(i)) = " << (ok ? "true" : "false") << " expected true\n";
  return ok;
}

}  // namespace

int main() {
  using namespace cute;

  std::cout << "CuTe layout algebra demo\n";

  bool ok = true;

  {
    std::cout << "\n[coalesce]\n";
    auto layout = Layout<Shape<_2, Shape<_1, _6>>, Stride<_1, Stride<_6, _2>>>{};
    auto result = coalesce(layout);
    auto by_mode = coalesce(layout, Step<_1, _1>{});

    print_object("layout       ", layout);
    print_object("coalesce     ", result);
    print_object("coalesce modes", by_mode);

    ok = expect_index("size(coalesce(layout))", size(result), size(layout)) && ok;
    ok = expect_same_1d_mapping("coalesce(layout)", result, layout) && ok;
    ok = expect_same_1d_mapping("coalesce(layout, Step<_1,_1>)", by_mode, layout) && ok;
  }

  {
    std::cout << "\n[composition]\n";
    auto a = make_layout(make_shape(Int<6>{}, Int<2>{}), make_stride(Int<8>{}, Int<2>{}));
    auto b = make_layout(make_shape(Int<4>{}, Int<3>{}), make_stride(Int<3>{}, Int<1>{}));
    auto r = composition(a, b);

    print_object("A", a);
    print_object("B", b);
    print_object("A o B", r);

    ok = expect_true("compatible(B, A o B)", bool(compatible(b, r))) && ok;
    ok = expect_composition_property("composition", a, b, r) && ok;
    ok = expect_index("(A o B)(1)", r(1), 24) && ok;
    ok = expect_index("(A o B)(11)", r(11), 42) && ok;
  }

  {
    std::cout << "\n[composition reshape]\n";
    auto vector_layout = Layout<Int<20>, _2>{};
    auto matrix_order = make_layout(make_shape(Int<5>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{}));
    auto matrix_layout = composition(vector_layout, matrix_order);

    print_object("20:2", vector_layout);
    print_object("(5,4):(4,1)", matrix_order);
    print_object("20:2 o (5,4):(4,1)", matrix_layout);

    ok = expect_composition_property("reshape composition", vector_layout, matrix_order, matrix_layout) && ok;
    ok = expect_index("matrix_layout(3,2)", matrix_layout(3, 2), 28) && ok;
  }

  {
    std::cout << "\n[by-mode composition]\n";
    auto a = make_layout(make_shape(12, make_shape(4, 8)), make_stride(59, make_stride(13, 1)));
    auto tiler = make_tile(Layout<_3, _4>{}, Layout<_8, _2>{});
    auto result = composition(a, tiler);
    auto shape_tiler = make_shape(Int<3>{}, Int<8>{});
    auto shape_result = composition(a, shape_tiler);

    print_object("A", a);
    print_object("tiler <3:4,8:2>", tiler);
    print_object("composition(A, tiler)", result);
    print_object("composition(A, Shape<3,8>)", shape_result);

    ok = expect_index("composition(A, tiler)(2,7)", result(2, 7), 501) && ok;
    ok = expect_index("composition(A, Shape<3,8>)(2,7)", shape_result(2, 7), a(2, 7)) && ok;
  }

  {
    std::cout << "\n[complement]\n";
    auto tile = Layout<_4, _2>{};
    auto rest = complement(tile, Int<24>{});
    auto completed = make_layout(tile, rest);

    print_object("tile 4:2", tile);
    print_object("complement(tile, 24)", rest);
    print_object("(tile, complement)", completed);

    ok = expect_index("cosize((tile, complement))", cosize(completed), 24) && ok;
    ok = expect_index("complement(tile,24)(1)", rest(1), 1) && ok;
    ok = expect_index("complement(tile,24)(5)", rest(5), 17) && ok;
  }

  {
    std::cout << "\n[logical_divide and zipped_divide]\n";
    auto layout_a =
        make_layout(make_shape(Int<4>{}, Int<2>{}, Int<3>{}), make_stride(Int<2>{}, Int<1>{}, Int<8>{}));
    auto tiler = Layout<_4, _2>{};
    auto divided = logical_divide(layout_a, tiler);
    auto zipped = zipped_divide(layout_a, tiler);
    auto tile_layout = composition(layout_a, tiler);

    print_object("A", layout_a);
    print_object("tiler B", tiler);
    print_object("logical_divide(A,B)", divided);
    print_object("zipped_divide(A,B)", zipped);
    print_object("composition(A,B)", tile_layout);

    ok = expect_true("compatible(B, layout<0>(logical_divide(A,B)))",
                     bool(compatible(tiler, layout<0>(divided)))) &&
         ok;
    ok = expect_same_1d_mapping("layout<0>(zipped_divide(A,B))", layout<0>(zipped), tile_layout) && ok;
    ok = expect_index("zipped_divide(A,B)(3,0)", zipped(3, 0), tile_layout(3)) && ok;
  }

  {
    std::cout << "\n[logical_product]\n";
    auto block = make_layout(make_shape(Int<2>{}, Int<2>{}), make_stride(Int<4>{}, Int<1>{}));
    auto repeat = Layout<_6, _1>{};
    auto product = logical_product(block, repeat);

    print_object("block A", block);
    print_object("repeat B", repeat);
    print_object("logical_product(A,B)", product);

    ok = expect_same_1d_mapping("layout<0>(logical_product(A,B))", layout<0>(product), block) && ok;
    ok = expect_true("compatible(B, layout<1>(logical_product(A,B)))",
                     bool(compatible(repeat, layout<1>(product)))) &&
         ok;
    ok = expect_index("logical_product(A,B)(3,5)", product(3, 5), product(make_coord(3, 5))) && ok;
  }

  {
    std::cout << "\n[blocked_product and raked_product]\n";
    auto block = make_layout(make_shape(Int<2>{}, Int<5>{}), make_stride(Int<5>{}, Int<1>{}));
    auto layout_of_blocks = make_layout(make_shape(Int<3>{}, Int<4>{}), make_stride(Int<1>{}, Int<3>{}));
    auto blocked = blocked_product(block, layout_of_blocks);
    auto raked = raked_product(block, layout_of_blocks);

    print_object("block", block);
    print_object("layout_of_blocks", layout_of_blocks);
    print_object("blocked_product", blocked);
    print_object("raked_product", raked);

    ok = expect_index("size(blocked_product)", size(blocked), size(block) * size(layout_of_blocks)) && ok;
    ok = expect_index("size(raked_product)", size(raked), size(block) * size(layout_of_blocks)) && ok;
  }

  if (!ok) {
    std::cerr << "layout algebra check failed\n";
    return EXIT_FAILURE;
  }

  std::cout << "\nlayout algebra check passed\n";
  return EXIT_SUCCESS;
}
