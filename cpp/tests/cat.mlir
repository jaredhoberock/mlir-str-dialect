// RUN: mlir-opt --convert-to-llvm %s | FileCheck %s

// CHECK-LABEL: llvm.func @test1
// CHECK: memcpy
// CHECK-NOT: builtin.unrealized_conversion_cast
func.func @test1(%a: !str.string, %b: !str.string) -> !str.string {
  %res = str.cat %a, %b
  return %res : !str.string
}
