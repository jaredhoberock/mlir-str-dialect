// RUN: mlir-opt --convert-to-llvm %s | FileCheck %s

// ---- Basic string constant
// CHECK-LABEL: llvm.func @const_string
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @const_string() -> !str.string {
  %str = str.constant "hello" : !str.string
  return %str : !str.string
}

// ---- Empty string constant
// CHECK-LABEL: llvm.func @empty_string
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @empty_string() -> !str.string {
  %str = str.constant "" : !str.string
  return %str : !str.string
}

// ---- String with special characters
// CHECK-LABEL: llvm.func @special_chars
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @special_chars() -> !str.string {
  %str = str.constant "hello\nworld\t\"escaped\"" : !str.string
  return %str : !str.string
}
