#ifndef STR_TYPES_TD
#define STR_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "Dialect.td"

// Represents the type of an immutable string
def StringType : TypeDef<StrDialect, "String"> {
  let mnemonic = "string";
  let summary = "Immutable string value";

  let description = [{
    Represents an immutable string.
  }];
}

#endif // STR_TYPES_TD
