#pragma once

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;

namespace str {

void populateStrToLLVMConversionPatterns(LLVMTypeConverter& typeConverter,
                                         RewritePatternSet& patterns);
}
}
