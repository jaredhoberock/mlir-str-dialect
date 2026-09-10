// RUN: mlir-opt -inline %s | FileCheck %s

// MLIR's inliner clones a callee only when every op in it belongs to a dialect
// registering an inliner interface. A function holding a str op is inlinable only
// because the str dialect registers one; without it the call below survives.
// After -inline @caller holds the str.constant directly and no call remains.

// CHECK-LABEL: func @caller
// CHECK-NOT: call
// CHECK: str.constant
// CHECK: return
func.func @caller() -> !str.string {
  %r = call @callee() : () -> !str.string
  return %r : !str.string
}

func.func private @callee() -> !str.string {
  %s = str.constant "hello" : !str.string
  return %s : !str.string
}
