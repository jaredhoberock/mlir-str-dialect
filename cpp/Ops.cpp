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
} // end mlir::str
