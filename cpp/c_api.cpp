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
  auto op = ConstantOp::create(builder, 
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
  auto op = CmpOp::create(builder, 
    unwrap(loc),
    cppPredicate,
    unwrap(lhs),
    unwrap(rhs)
  );
  return wrap(op.getOperation());
}

MlirOperation strAsMemRefOpCreate(MlirLocation loc,
                                  MlirValue input) {
  MLIRContext *ctx = unwrap(loc).getContext();
  OpBuilder builder(ctx);

  auto op = AsMemRefOp::create(builder, 
    unwrap(loc),
    unwrap(input)
  );
  return wrap(op.getOperation());
}

MlirOperation strCatOpCreate(MlirLocation loc,
                             MlirValue lhs,
                             MlirValue rhs) {
  MLIRContext *ctx = unwrap(loc).getContext();
  OpBuilder builder(ctx);

  auto op = CatOp::create(builder, 
    unwrap(loc),
    unwrap(lhs),
    unwrap(rhs)
  );
  return wrap(op.getOperation());
}

MlirOperation strFormatOpCreate(MlirLocation loc,
                                MlirValue format,
                                const MlirValue *args, intptr_t numArgs) {
  MLIRContext *ctx = unwrap(loc).getContext();
  OpBuilder builder(ctx);
  SmallVector<Value> argValues;
  for (intptr_t i = 0; i < numArgs; ++i) {
    argValues.push_back(unwrap(args[i]));
  }
  auto op = FormatOp::create(builder, 
    unwrap(loc),
    unwrap(format),
    argValues
  );
  return wrap(op.getOperation());
}

} // end extern "C"
