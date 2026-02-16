// RUN: mlir-opt --convert-to-llvm %s | FileCheck %s

// ---- Format a single i64
// CHECK-LABEL: llvm.func @format_i64
// CHECK: llvm.call @snprintf
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @format_i64(%x: i64) -> !str.string {
  %fmt = str.constant "%lld" : !str.string
  %s = str.format %fmt(%x) : (!str.string, i64) -> !str.string
  return %s : !str.string
}

// ---- Format multiple i64 values
// CHECK-LABEL: llvm.func @format_two_i64
// CHECK: llvm.call @snprintf
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @format_two_i64(%a: i64, %b: i64) -> !str.string {
  %fmt = str.constant "(%lld, %lld)" : !str.string
  %s = str.format %fmt(%a, %b) : (!str.string, i64, i64) -> !str.string
  return %s : !str.string
}

// ---- Format a bool (i1)
// CHECK-LABEL: llvm.func @format_bool
// CHECK: llvm.zext
// CHECK: llvm.call @snprintf
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @format_bool(%x: i1) -> !str.string {
  %fmt = str.constant "%d" : !str.string
  %s = str.format %fmt(%x) : (!str.string, i1) -> !str.string
  return %s : !str.string
}

// ---- Format with no args (plain string)
// CHECK-LABEL: llvm.func @format_no_args
// CHECK: llvm.call @snprintf
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @format_no_args() -> !str.string {
  %fmt = str.constant "hello" : !str.string
  %s = str.format %fmt() : (!str.string) -> !str.string
  return %s : !str.string
}
