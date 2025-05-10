#include "Dialect.hpp"
#include "Lowering.hpp"
#include "Ops.hpp"
#include "Types.hpp"
#include <mlir/Conversion/LLVMCommon/MemRefBuilder.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::str {

struct ConstantOpLowering : OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  // cache mapping string hash to symbol name
  mutable llvm::DenseMap<uint64_t, StringAttr> globalSymbolCache;

  LogicalResult matchAndRewrite(
      ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    std::string stringValue = op.getValue().str();
    stringValue.push_back('\0'); // ensure null termination

    uint64_t hash = llvm::hash_value(stringValue);
    StringAttr symbol;

    // create types
    auto i8Type = IntegerType::get(rewriter.getContext(), 8);
    int64_t length = static_cast<int64_t>(stringValue.size());
    auto staticType = MemRefType::get({length}, i8Type);
    auto dynamicType = MemRefType::get({ShapedType::kDynamic}, i8Type);

    // get or create the memref.global
    ModuleOp module = op->getParentOfType<ModuleOp>();

    // check cache first
    auto it = globalSymbolCache.find(hash);
    if (it == globalSymbolCache.end()) {
      // create a name for the global based on the hash
      std::string globalName = "__str_" + std::to_string(hash);
      symbol = rewriter.getStringAttr(globalName);
      globalSymbolCache[hash] = symbol;

      // construct DenseElementsAttr from characters
      SmallVector<APInt,16> chars;
      chars.reserve(length);
      for (uint8_t c : stringValue) {
        chars.push_back(APInt(8,c));
      }

      auto tensorType = RankedTensorType::get({length}, i8Type);
      auto initialValueAttr = DenseElementsAttr::get(tensorType, chars);

      // insert the global at module scope
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<memref::GlobalOp>(
        op.getLoc(),
        rewriter.getStringAttr(globalName), // sym_name
        rewriter.getStringAttr("private"),  // sym_visibility
        staticType,                         // type
        initialValueAttr,                   // initial_value
        true,                               // constant
        nullptr                             // alignment - use nullptr for default
      );
    } else {
      symbol = it->second;
    }

    // create reference to the global
    Value global = rewriter.create<memref::GetGlobalOp>(
      op.getLoc(), staticType, symbol);

    // cast to dynamic type and replace the original op
    rewriter.replaceOpWithNewOp<memref::CastOp>(
      op, dynamicType, global);

    return success();
  }
};

struct CmpOpLowering : OpConversionPattern<CmpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CmpOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = rewriter.getContext();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Type i32Type = rewriter.getI32Type();

    // get pointer type for strcmp
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    // declare strcmp if needed
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("strcmp")) {
      auto strcmpType = LLVM::LLVMFunctionType::get(i32Type, {ptrType, ptrType}, false);
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(loc, "strcmp", strcmpType);
    }

    // extract aligned pointers from memref descriptors
    MemRefDescriptor lhsDesc(lhs);
    MemRefDescriptor rhsDesc(rhs);
    Value lhsPtr = lhsDesc.alignedPtr(rewriter, loc);
    Value rhsPtr = rhsDesc.alignedPtr(rewriter, loc);

    // call strcmp
    Value result = rewriter.create<LLVM::CallOp>(
      loc,
      i32Type,
      "strcmp",
      ValueRange{lhsPtr, rhsPtr}
    ).getResult();

    // compare against zero
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);

    arith::CmpIPredicate pred;
    switch (op.getPredicate()) {
      case CmpPredicate::eq: pred = arith::CmpIPredicate::eq; break;
      case CmpPredicate::ne: pred = arith::CmpIPredicate::ne; break;
      case CmpPredicate::lt: pred = arith::CmpIPredicate::slt; break;
      case CmpPredicate::le: pred = arith::CmpIPredicate::sle; break;
      case CmpPredicate::gt: pred = arith::CmpIPredicate::sgt; break;
      case CmpPredicate::ge: pred = arith::CmpIPredicate::sge; break;
    }

    rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, pred, result, zero);
    return success();
  }
};

struct AsMemRefOpLowering : OpConversionPattern<str::AsMemRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      str::AsMemRefOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // we assume that the input has already been lowered to memref
    // (or something lower than memref, such as llvm.struct)
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

void populateStrToLLVMConversionPatterns(LLVMTypeConverter& typeConverter, RewritePatternSet& patterns) {
  // add conversion from !str.string to memref<?xi8>
  typeConverter.addConversion([&](StringType type) -> Type {
    Type memrefType = MemRefType::get({ShapedType::kDynamic}, IntegerType::get(type.getContext(), 8));

    // XXX this seems like a bug
    //     it shouldn't be necessary to apply the type converter to the intermediate type
    //     MLIR should handle that for us automatically
    return typeConverter.convertType(memrefType);
  });

  patterns.add<
    AsMemRefOpLowering,
    CmpOpLowering,
    ConstantOpLowering
  >(typeConverter, patterns.getContext());

  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
}

}
