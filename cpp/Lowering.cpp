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
#include <mlir/Dialect/MemRef/Transforms/Transforms.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::str {

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

struct CatOpLowering : OpConversionPattern<CatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CatOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // get inputs as memrefs
    Value lhs = AsMemRefOp::create(rewriter, loc, op.getLhs());
    Value rhs = AsMemRefOp::create(rewriter, loc, op.getRhs());

    // get lengths including null
    Value lhsSize = memref::DimOp::create(rewriter, loc, lhs, 0);
    Value rhsSize = memref::DimOp::create(rewriter, loc, rhs, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);

    // subtract one for the null terminator
    Value lhsLen = arith::SubIOp::create(rewriter, loc, lhsSize, one);
    Value rhsLen = arith::SubIOp::create(rewriter, loc, rhsSize, one);

    // total length + null
    Value sumLen = arith::AddIOp::create(rewriter, loc, lhsLen, rhsLen);
    Value totalSize = arith::AddIOp::create(rewriter, loc, sumLen, one);

    // alloc result memref<?xi8>
    auto memrefTy = MemRefType::get({ShapedType::kDynamic}, rewriter.getI8Type());
    Value alloc = memref::AllocOp::create(rewriter, loc, memrefTy, totalSize);

    // copy lhs region
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto lhsDst = memref::SubViewOp::create(rewriter, loc, memrefTy, alloc,
      ValueRange{zero}, ValueRange{lhsLen}, ValueRange{one});
    auto lhsSrc = memref::SubViewOp::create(rewriter, loc, memrefTy, lhs,
      ValueRange{zero}, ValueRange{lhsLen}, ValueRange{one});
    memref::CopyOp::create(rewriter, loc, lhsSrc, lhsDst);

    // copy rhs region
    auto rhsDst = memref::SubViewOp::create(rewriter, loc, memrefTy, alloc,
      ValueRange{lhsLen}, ValueRange{rhsLen}, ValueRange{one});
    auto rhsSrc = memref::SubViewOp::create(rewriter, loc, memrefTy, rhs,
      ValueRange{zero}, ValueRange{rhsLen}, ValueRange{one});
    memref::CopyOp::create(rewriter, loc, rhsSrc, rhsDst);

    // null-terminate
    Value zero8 = arith::ConstantIntOp::create(rewriter, loc, 0, 8);
    memref::StoreOp::create(rewriter, 
      loc,
      zero8,
      alloc,
      ValueRange{sumLen});

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

struct ConstantOpLowering : OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  // cache mapping (symbol table, string hash) to symbol name
  mutable llvm::DenseMap<std::pair<Operation*, uint64_t>, StringAttr> globalSymbolCache;

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

    // find nearest symbol table
    Operation *symbolTable = SymbolTable::getNearestSymbolTable(op);
    if (!symbolTable)
      return rewriter.notifyMatchFailure(op, "not inside a symbol table");

    // cache key includes symbol table
    auto cacheKey = std::make_pair(symbolTable, hash);

    // check cache first
    auto it = globalSymbolCache.find(cacheKey);
    if (it == globalSymbolCache.end()) {
      // create a name for the global based on the hash
      std::string globalName = "__str_" + std::to_string(hash);
      symbol = rewriter.getStringAttr(globalName);
      globalSymbolCache[cacheKey] = symbol;

      // construct DenseElementsAttr from characters
      SmallVector<APInt,16> chars;
      chars.reserve(length);
      for (uint8_t c : stringValue) {
        chars.push_back(APInt(8,c));
      }

      auto tensorType = RankedTensorType::get({length}, i8Type);
      auto initialValueAttr = DenseElementsAttr::get(tensorType, chars);

      // insert the global at symbol table scope
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&symbolTable->getRegion(0).front());
      memref::GlobalOp::create(rewriter, 
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
    Value global = memref::GetGlobalOp::create(rewriter, 
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
      LLVM::LLVMFuncOp::create(rewriter, loc, "strcmp", strcmpType);
    }

    // extract aligned pointers from memref descriptors
    MemRefDescriptor lhsDesc(lhs);
    MemRefDescriptor rhsDesc(rhs);
    Value lhsPtr = lhsDesc.alignedPtr(rewriter, loc);
    Value rhsPtr = rhsDesc.alignedPtr(rewriter, loc);

    // call strcmp
    Value result = LLVM::CallOp::create(rewriter, 
      loc,
      i32Type,
      "strcmp",
      ValueRange{lhsPtr, rhsPtr}
    ).getResult();

    // compare against zero
    Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);

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

/// Lowers str.format to snprintf via a measure-then-write pattern.
///
/// The format string is already lowered to a memref descriptor by ConstantOpLowering.
/// We call snprintf twice: once to measure the output length, once to write it.
/// The result buffer is allocated via memref.alloc so it's visible to future
/// deallocation passes.
///
/// Input:
///   %fmt = str.constant "%lld" : !str.string
///   %s = str.format %fmt(%x) : (!str.string, i64) -> !str.string
///
/// Lowers to (approximately):
///   // measure
///   %len = llvm.call @snprintf(nullptr, 0, %fmt_ptr, %x) : ... -> i32
///   %buf_size = llvm.add %len, 1
///
///   // allocate
///   %buf = memref.alloc(%buf_size) : memref<?xi8>
///   %buf_ptr = <extract pointer from %buf>
///
///   // write
///   llvm.call @snprintf(%buf_ptr, %buf_size, %fmt_ptr, %x) : ...
///
/// Notes:
/// - i1 arguments are promoted to i32 per C variadic calling convention
/// - snprintf is declared as a variadic LLVM function because func.call
///   does not support varargs
/// - The format string pointer is extracted from the already-lowered memref
///   descriptor, same as CmpOpLowering does for strcmp
struct FormatOpLowering : OpConversionPattern<FormatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      FormatOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    Type i8Ty = rewriter.getI8Type();
    Type i32Ty = rewriter.getI32Type();
    Type i64Ty = rewriter.getI64Type();
    auto memrefTy = MemRefType::get({ShapedType::kDynamic}, i8Ty);
    auto snprintfTy = LLVM::LLVMFunctionType::get(
      i32Ty,
      {ptrTy, i64Ty, ptrTy},
      /*isVarArg=*/true
    );

    // declare snprintf if needed
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("snprintf")) {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      LLVM::LLVMFuncOp::create(rewriter, loc, "snprintf", snprintfTy);
    }

    // extract format string pointer from lowered memref descriptor
    MemRefDescriptor fmtDesc(adaptor.getFormat());
    Value fmtPtr = fmtDesc.alignedPtr(rewriter, loc);

    // promote i1 args to i32 for C variadic calling convention
    SmallVector<Value> fmtArgs;
    for (Value arg : adaptor.getArgs()) {
      if (arg.getType().isInteger(1)) {
        fmtArgs.push_back(
          LLVM::ZExtOp::create(rewriter, loc, i32Ty, arg));
      } else {
        fmtArgs.push_back(arg);
      }
    }

    // step 1: measure — snprintf(nullptr, 0, fmt, args...) → length
    Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
    Value zero64 = LLVM::ConstantOp::create(rewriter, 
      loc, i64Ty, rewriter.getI64IntegerAttr(0));

    SmallVector<Value> measureArgs = {nullPtr, zero64, fmtPtr};
    measureArgs.append(fmtArgs.begin(), fmtArgs.end());

    Value len32 = LLVM::CallOp::create(rewriter, 
      loc,
      snprintfTy,
      "snprintf",
      measureArgs
    ).getResult();

    // bufSize = len + 1 (for null terminator)
    Value one32 = LLVM::ConstantOp::create(rewriter, 
      loc, i32Ty, rewriter.getI32IntegerAttr(1));
    Value bufSize32 = LLVM::AddOp::create(rewriter, loc, len32, one32);
    Value bufSize = LLVM::SExtOp::create(rewriter, loc, i64Ty, bufSize32);

    // convert to index for memref.alloc
    Value bufSizeIdx = arith::IndexCastOp::create(rewriter, 
      loc, rewriter.getIndexType(), bufSize);

    // step 2: allocate buffer via memref.alloc
    Value alloc = memref::AllocOp::create(rewriter, loc, memrefTy, bufSizeIdx);

    // extract pointer from memref for snprintf
    Value ptrIdx = memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc, alloc);
    Value ptrInt = arith::IndexCastOp::create(rewriter, loc, i64Ty, ptrIdx);
    Value bufPtr = LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, ptrInt);

    // step 3: format — snprintf(buf, bufSize, fmt, args...)
    SmallVector<Value> writeArgs = {bufPtr, bufSize, fmtPtr};
    writeArgs.append(fmtArgs.begin(), fmtArgs.end());
    LLVM::CallOp::create(rewriter, 
      loc,
      snprintfTy,
      "snprintf",
      writeArgs
    );

    // step 4: replace with the memref — type conversion handles the rest
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateStrToLLVMConversionPatterns(LLVMTypeConverter& typeConverter, RewritePatternSet& patterns) {
  // add conversion from !str.string to memref<?xi8>
  typeConverter.addConversion([&](StringType type) -> Type {
    Type memrefType = MemRefType::get({ShapedType::kDynamic}, IntegerType::get(type.getContext(), 8));

    // recurse on the memref type to convert to an LLVM type
    return typeConverter.convertType(memrefType);
  });

  patterns.add<
    AsMemRefOpLowering,
    CatOpLowering,
    CmpOpLowering,
    ConstantOpLowering,
    FormatOpLowering
  >(typeConverter, patterns.getContext());

  memref::populateExpandStridedMetadataPatterns(patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
}

}
