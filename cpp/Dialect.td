#ifndef STR_DIALECT_TD
#define STR_DIALECT_TD

include "mlir/IR/OpBase.td"

def StrDialect : Dialect {
  let name = "str";
  let summary = "A dialect for operations on immutable strings";
  let cppNamespace = "::mlir::str";

  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::func::FuncDialect",
  ];

  let hasCanonicalizer = 1;
}

class StrOp<string mnemonic, list<Trait> traits = []> :
  Op<StrDialect, mnemonic, traits>;

#endif // STR_DIALECT_TD
