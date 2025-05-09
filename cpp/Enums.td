#ifndef STR_ENUMS_TD
#define STR_ENUMS_TD

include "mlir/IR/EnumAttr.td"
include "Dialect.td"

def CmpPredicateAttr : I64EnumAttr<
  "CmpPredicate", "",
  [
    I64EnumAttrCase<"eq", 0>,
    I64EnumAttrCase<"ne", 1>,
    I64EnumAttrCase<"lt", 2>,
    I64EnumAttrCase<"le", 3>,
    I64EnumAttrCase<"gt", 4>,
    I64EnumAttrCase<"ge", 5>
  ]
> {
  let cppNamespace = "::mlir::str";
}

#endif // STR_ENUMS_TD
