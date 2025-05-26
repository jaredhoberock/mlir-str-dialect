#ifndef STR_OPS_TD
#define STR_OPS_TD

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "Dialect.td"
include "Enums.td"
include "Types.td"

def ConstantOp : StrOp<"constant", [Pure]> {
  let summary = "Create a string constant";
  let description = [{
    Materializes a compile-time string literal of type `!str.string`.

    Example:
      %s = str.constant "hello" : !str.string
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs StringType);

  let builders = [
    OpBuilder<(ins "StringRef":$value), [{
      $_state.addAttribute("value", $_builder.getStringAttr(value));
      $_state.addTypes($_builder.getType<::mlir::str::StringType>());
    }]>
  ];

  let assemblyFormat = "$value attr-dict `:` type(results)";
}

def CmpOp : StrOp<"cmp", [Pure]> {
  let summary = "Compare two `!str.string` values";
  let description = [{
    Compares two immutable string values using a comparison predicate.
    Returns an `i1` indicating the result.

    Example:
      %cmp = str.cmp lt, %a, %b : !str.string
  }];

  let arguments = (ins
    CmpPredicateAttr:$predicate,
    StringType:$lhs,
    StringType:$rhs
  );

  let results = (outs I1);

  let builders = [
    OpBuilder<(ins "CmpPredicate":$predicate, "Value":$lhs, "Value":$rhs), [{
      build($_builder, $_state,
            $_builder.getI1Type(),
            $_builder.getAttr<::mlir::str::CmpPredicateAttr>(predicate),
            lhs, rhs);
    }]>
  ];

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

def CatOp : StrOp<"cat", [Pure]> {
  let summary = "Concatenate two `!str.string` values";
  let description = [{
    Concatenates two `!str.string` values into a new string.

    Example:
      %c = str.cat %a, %b : !str.string
  }];

  let arguments = (ins
    StringType:$lhs,
    StringType:$rhs
  );

  let results = (outs StringType);

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
      build($_builder, $_state,
            $_builder.getType<::mlir::str::StringType>(),
            lhs, rhs);
    }]>
  ];

  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def AsMemRefOp : StrOp<"as_memref", [Pure]> {
  let summary = "Expose the underlying memref of a string";
  let description = [{
    Converts a `!str.string` into a `memref<?xi8>` referencing its contents.
    
    This operation is useful when interfacing with lower-level dialects like `printf`,
    which expect a raw character buffer.

    Example:
      %buffer = str.as_memref %s : !str.string to memref<?xi8>
  }];

  let arguments = (ins StringType:$input);
  let results = (outs AnyMemRef:$output);

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$input), [{
      auto memrefType = MemRefType::get({ShapedType::kDynamic}, $_builder.getI8Type());
      build($_builder, $_state, memrefType, input);
    }]>
  ];

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

#endif // STR_OPS_TD
