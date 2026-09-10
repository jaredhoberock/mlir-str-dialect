#include "Dialect.hpp"
#include "Ops.hpp"
#include "Lowering.hpp"
#include "Types.hpp"
#include <llvm/ADT/STLExtras.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Transforms/InliningUtils.h>

#include <Dialect.cpp.inc>

namespace mlir::str {

struct ConvertToLLVMInterface : public mlir::ConvertToLLVMPatternInterface {
  using mlir::ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(ConversionTarget& target,
                                               LLVMTypeConverter& typeConverter,
                                               RewritePatternSet& patterns) const override final {
    populateStrToLLVMConversionPatterns(typeConverter, patterns);
  }
};

/// Declares inlining legal for the str dialect. MLIR's inliner clones a callee
/// only when every op in it belongs to a dialect registering this interface, so a
/// function holding a str op is inlinable only once str provides one. The hooks
/// are permissive: str's ops carry no state that forbids cloning them into
/// another region.
struct StrInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }
};

void StrDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include <Ops.cpp.inc>
  >();

  registerTypes();

  addInterfaces<
    ConvertToLLVMInterface,
    StrInlinerInterface
  >();
}

}
