// RUN: mlir-opt %s -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func @cat_two_constants
// CHECK: str.constant "hello, world"
func.func @cat_two_constants() -> !str.string {
  %a = str.constant "hello, " : !str.string
  %b = str.constant "world" : !str.string
  %res = str.cat %a, %b
  return %res : !str.string
}
