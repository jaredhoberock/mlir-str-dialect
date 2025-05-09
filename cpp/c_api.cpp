#include "c_api.h"
#include "Dialect.hpp"
#include "Enums.hpp"
#include "Ops.hpp"
#include "Types.hpp"
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/IR/Builders.h>

using namespace mlir;
using namespace mlir::str;

extern "C" {

void strRegisterDialect(MlirContext context) {
  unwrap(context)->loadDialect<StrDialect>();
}

MlirType strStringTypeGet(MlirContext context) {
  return wrap(StringType::get(unwrap(context)));
}

MlirOperation strConstantOpCreate(MlirLocation loc,
                                  MlirStringRef value) {
  MLIRContext *ctx = unwrap(loc).getContext();
  OpBuilder builder(ctx);
  auto op = builder.create<ConstantOp>(
    unwrap(loc),
    StringRef(value.data, value.length)
  );
  return wrap(op.getOperation());
}

MlirOperation strCmpOpCreate(MlirLocation loc,
                             StrCmpPredicate predicate,
                             MlirValue lhs,
                             MlirValue rhs) {
  MLIRContext *ctx = unwrap(loc).getContext();
  OpBuilder builder(ctx);

  auto cppPredicate = static_cast<str::CmpPredicate>(predicate);
  auto op = builder.create<CmpOp>(
    unwrap(loc),
    cppPredicate,
    unwrap(lhs),
    unwrap(rhs)
  );
  return wrap(op.getOperation());
}

} // end extern "C"
