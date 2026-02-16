// RUN: mlir-opt --convert-to-llvm %s \
// RUN:   | mlir-translate --mlir-to-llvmir \
// RUN:   | opt -O2 \
// RUN:   | llc -relocation-model=pic -filetype=obj -o %t.o
// RUN: clang %t.o -o %t
// RUN: %t

func.func @main() -> i32 {
  %fmt = str.constant "%lld" : !str.string
  %x = arith.constant 42 : i64
  %s = str.format %fmt(%x) : (!str.string, i64) -> !str.string

  %expected = str.constant "42" : !str.string
  %eq = str.cmp eq, %s, %expected : !str.string

  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %ret = arith.select %eq, %c0, %c1 : i1, i32
  return %ret : i32
}
