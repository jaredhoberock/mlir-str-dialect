#include "Dialect.hpp"
#include "Ops.hpp"
#include <mlir/IR/PatternMatch.h>

namespace mlir::str {

struct FoldCatConstants : public OpRewritePattern<CatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CatOp op,
                                PatternRewriter& rewriter) const override {
    // check if both inputs are constant
    auto lhsDefOp = op.getLhs().getDefiningOp<ConstantOp>();
    auto rhsDefOp = op.getRhs().getDefiningOp<ConstantOp>();

    if (not lhsDefOp or not rhsDefOp)
      return failure();

    StringRef c1 = lhsDefOp.getValue();
    StringRef c2 = rhsDefOp.getValue();
    std::string concatenated = (c1 + c2).str();

    rewriter.replaceOpWithNewOp<ConstantOp>(
      op,
      op.getType(),
      rewriter.getStringAttr(concatenated)
    );
    return success();
  }
};

void StrDialect::getCanonicalizationPatterns(RewritePatternSet& patterns) const {
  patterns.add<FoldCatConstants>(patterns.getContext());
}

} // end mlir::str
