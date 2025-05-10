#pragma once

#include <mlir-c/IR.h>
#include <mlir-c/Pass.h>
#include <mlir-c/Support.h>

#ifdef __cplusplus
extern "C" {
#endif

void strRegisterDialect(MlirContext ctx);

MlirType strStringTypeGet(MlirContext ctx);

MlirOperation strConstantOpCreate(MlirLocation loc,
                                  MlirStringRef value);

typedef enum {
  StrCmpPredicateEq = 0,
  StrCmpPredicateNe,
  StrCmpPredicateLt,
  StrCmpPredicateLe,
  StrCmpPredicateGt,
  StrCmpPredicateG3
} StrCmpPredicate;

MlirOperation strCmpOpCreate(MlirLocation loc,
                             StrCmpPredicate predicate,
                             MlirValue lhs,
                             MlirValue rhs);

MlirOperation strAsMemRefOpCreate(MlirLocation loc,
                                  MlirValue input);

#ifdef __cplusplus
}
#endif
