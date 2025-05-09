#pragma once

#include <llvm/Support/Error.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include "Enums.hpp"
#include "Types.hpp"

#define GET_OP_CLASSES
#include "Ops.hpp.inc"
