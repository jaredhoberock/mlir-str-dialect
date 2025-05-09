// RUN: opt --convert-to-llvm %s | FileCheck %s

// ---- Test 1: Basic string constant
// CHECK-LABEL: llvm.func @const_string
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @const_string() -> !str.string {
  %str = str.constant "hello" : !str.string
  return %str : !str.string
}

// ---- Test 2: Empty string constant
// CHECK-LABEL: llvm.func @empty_string
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @empty_string() -> !str.string {
  %str = str.constant "" : !str.string
  return %str : !str.string
}

// ---- Test 3: String with special characters
// CHECK-LABEL: llvm.func @special_chars
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @special_chars() -> !str.string {
  %str = str.constant "hello\nworld\t\"escaped\"" : !str.string
  return %str : !str.string
}

// ---- Test 4: str.cmp eq
// CHECK-LABEL: llvm.func @str_cmp_eq
// CHECK: llvm.call @strcmp
// CHECK: llvm.icmp "eq"
func.func @str_cmp_eq(%a: !str.string, %b: !str.string) -> i1 {
  %res = str.cmp eq, %a, %b : !str.string
  return %res : i1
}

// ---- Test 4.5 Two identical string constants
// CHECK-LABEL: llvm.func @two_identical_strings
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @two_identical_strings() -> i1 {
  %a = str.constant "hello" : !str.string
  %b = str.constant "hello" : !str.string
  %res = str.cmp eq, %a, %b : !str.string
  return %res : i1
}

// ---- Test 5: str.cmp ne
// CHECK-LABEL: llvm.func @str_cmp_ne
// CHECK: llvm.call @strcmp
// CHECK: llvm.icmp "ne"
func.func @str_cmp_ne(%a: !str.string, %b: !str.string) -> i1 {
  %res = str.cmp ne, %a, %b : !str.string
  return %res : i1
}

// ---- Test 6: str.cmp lt
// CHECK-LABEL: llvm.func @str_cmp_lt
// CHECK: llvm.call @strcmp
// CHECK: llvm.icmp "slt"
func.func @str_cmp_lt(%a: !str.string, %b: !str.string) -> i1 {
  %res = str.cmp lt, %a, %b : !str.string
  return %res : i1
}

// ---- Test 6: str.cmp le
// CHECK-LABEL: llvm.func @str_cmp_le
// CHECK: llvm.call @strcmp
// CHECK: llvm.icmp "sle"
func.func @str_cmp_le(%a: !str.string, %b: !str.string) -> i1 {
  %res = str.cmp le, %a, %b : !str.string
  return %res : i1
}

// ---- Test 7: str.cmp gt
// CHECK-LABEL: llvm.func @str_cmp_gt
// CHECK: llvm.call @strcmp
// CHECK: llvm.icmp "sgt"
func.func @str_cmp_gt(%a: !str.string, %b: !str.string) -> i1 {
  %res = str.cmp gt, %a, %b : !str.string
  return %res : i1
}

// ---- Test 8: str.cmp ge
// CHECK-LABEL: llvm.func @str_cmp_ge
// CHECK: llvm.call @strcmp
// CHECK: llvm.icmp "sge"
func.func @str_cmp_ge(%a: !str.string, %b: !str.string) -> i1 {
  %res = str.cmp ge, %a, %b : !str.string
  return %res : i1
}
