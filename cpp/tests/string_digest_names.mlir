// RUN: mlir-opt --convert-to-llvm %s | FileCheck %s

// The global's name is a stable content digest of the string's bytes, so the same
// string names one global whatever the process: "hello" names one global across
// the two functions that hold it, "world" names another.

// CHECK-DAG: llvm.mlir.global private constant @__str_c5ab198c53eb7646(
// CHECK-DAG: llvm.mlir.global private constant @__str_8320a96a612150a7(
func.func @first() -> !str.string {
  %s = str.constant "hello" : !str.string
  return %s : !str.string
}
func.func @second() -> !str.string {
  %s = str.constant "hello" : !str.string
  return %s : !str.string
}
func.func @third() -> !str.string {
  %s = str.constant "world" : !str.string
  return %s : !str.string
}
