use melior::{ir::{Location, Operation, Type, Value, ValueLike}, Context};
use mlir_sys::{MlirContext, MlirLocation, MlirOperation, MlirStringRef, MlirType, MlirValue};

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum CmpPredicate {
    Eq = 0,
    Ne = 1,
    Lt = 2,
    Le = 3,
    Gt = 4,
    Ge = 5,
}

unsafe extern "C" {
    fn strRegisterDialect(ctx: MlirContext);
    fn strStringTypeGet(ctx: MlirContext) -> MlirType;
    fn strConstantOpCreate(loc: MlirLocation, value: MlirStringRef) -> MlirOperation;
    fn strCmpOpCreate(loc: MlirLocation, predicate: CmpPredicate, lhs: MlirValue, rhs: MlirValue) -> MlirOperation;
    fn strAsMemRefOpCreate(loc: MlirLocation, input: MlirValue) -> MlirOperation;
}

pub fn register(context: &Context) {
    unsafe { strRegisterDialect(context.to_raw()) }
}

pub fn string_type(context: &Context) -> Type {
    unsafe { Type::from_raw(strStringTypeGet(context.to_raw())) }
}

pub fn constant<'c>(
    location: Location<'c>,
    value: &str,
) -> Operation<'c> {
    unsafe {
        let raw_op = strConstantOpCreate(
            location.to_raw(),
            MlirStringRef {
                data: value.as_ptr().cast(),
                length: value.len(),
            },
        );
        Operation::from_raw(raw_op)
    }
}

pub fn cmp<'c>(
    loc: Location<'c>,
    pred: CmpPredicate,
    lhs: Value<'c,'_>,
    rhs: Value<'c,'_>,
) -> Operation<'c> {
    unsafe {
        Operation::from_raw(strCmpOpCreate(
            loc.to_raw(),
            pred,
            lhs.to_raw(),
            rhs.to_raw(),
        ))
    }
}

pub fn as_memref<'c>(loc: Location<'c>, input: Value<'c,'_>) -> Operation<'c> {
    unsafe {
        Operation::from_raw(strAsMemRefOpCreate(
            loc.to_raw(),
            input.to_raw(),
        ))
    }
}
