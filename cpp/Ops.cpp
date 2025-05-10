#include "Dialect.hpp"
#include "Ops.hpp"
#include "Types.hpp"
#include <mlir/IR/Builders.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <iostream>

#define GET_OP_CLASSES
#include "Ops.cpp.inc"

namespace mlir::str {

LogicalResult str::AsMemRefOp::verify() {
  // Check that the result is a MemRef type
  Type resultType = getOutput().getType();
  if (!isa<MemRefType>(resultType)) {
    return emitOpError("result must be a memref type");
  }
  
  auto memrefType = cast<MemRefType>(resultType);
  
  // Check that the element type is i8
  if (!memrefType.getElementType().isInteger(8)) {
    return emitOpError("result must be a memref of i8 elements");
  }
  
  // Check that the memref has rank 1 with a dynamic dimension
  if (memrefType.getRank() != 1 || !memrefType.isDynamicDim(0)) {
    return emitOpError("result must be a memref<?xi8>");
  }
  
  return success();
}

} // end mlir::str
