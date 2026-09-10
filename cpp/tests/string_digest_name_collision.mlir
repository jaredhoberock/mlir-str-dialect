// RUN: mlir-opt --convert-to-llvm %s | FileCheck %s

// A global already carrying the digest name of "hello" but holding other bytes
// makes the lowering mint a fresh suffixed name rather than redefine the symbol,
// so both globals stand and the module lowers.

// CHECK-DAG: llvm.mlir.global private constant @__str_c5ab198c53eb7646(dense<[1, 2, 3]>
// CHECK-DAG: llvm.mlir.global private constant @__str_c5ab198c53eb7646_0(dense<[104, 101, 108, 108, 111, 0]>
memref.global "private" constant @__str_c5ab198c53eb7646 : memref<3xi8> = dense<[1, 2, 3]>

func.func @use() -> !str.string {
  %s = str.constant "hello" : !str.string
  return %s : !str.string
}
