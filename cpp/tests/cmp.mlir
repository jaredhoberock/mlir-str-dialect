// RUN: mlir-opt --convert-to-llvm %s | FileCheck %s

// ---- str.cmp eq
// CHECK-LABEL: llvm.func @str_cmp_eq
// CHECK: llvm.call @strcmp
// CHECK: llvm.icmp "eq"
func.func @str_cmp_eq(%a: !str.string, %b: !str.string) -> i1 {
  %res = str.cmp eq, %a, %b : !str.string
  return %res : i1
}

// ---- Two identical string constants
// CHECK-LABEL: llvm.func @two_identical_strings
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @two_identical_strings() -> i1 {
  %a = str.constant "hello" : !str.string
  %b = str.constant "hello" : !str.string
  %res = str.cmp eq, %a, %b : !str.string
  return %res : i1
}

// ---- str.cmp ne
// CHECK-LABEL: llvm.func @str_cmp_ne
// CHECK: llvm.call @strcmp
// CHECK: llvm.icmp "ne"
func.func @str_cmp_ne(%a: !str.string, %b: !str.string) -> i1 {
  %res = str.cmp ne, %a, %b : !str.string
  return %res : i1
}

// ---- str.cmp lt
// CHECK-LABEL: llvm.func @str_cmp_lt
// CHECK: llvm.call @strcmp
// CHECK: llvm.icmp "slt"
func.func @str_cmp_lt(%a: !str.string, %b: !str.string) -> i1 {
  %res = str.cmp lt, %a, %b : !str.string
  return %res : i1
}

// ---- str.cmp le
// CHECK-LABEL: llvm.func @str_cmp_le
// CHECK: llvm.call @strcmp
// CHECK: llvm.icmp "sle"
func.func @str_cmp_le(%a: !str.string, %b: !str.string) -> i1 {
  %res = str.cmp le, %a, %b : !str.string
  return %res : i1
}

// ---- str.cmp gt
// CHECK-LABEL: llvm.func @str_cmp_gt
// CHECK: llvm.call @strcmp
// CHECK: llvm.icmp "sgt"
func.func @str_cmp_gt(%a: !str.string, %b: !str.string) -> i1 {
  %res = str.cmp gt, %a, %b : !str.string
  return %res : i1
}

// ---- str.cmp ge
// CHECK-LABEL: llvm.func @str_cmp_ge
// CHECK: llvm.call @strcmp
// CHECK: llvm.icmp "sge"
func.func @str_cmp_ge(%a: !str.string, %b: !str.string) -> i1 {
  %res = str.cmp ge, %a, %b : !str.string
  return %res : i1
}
