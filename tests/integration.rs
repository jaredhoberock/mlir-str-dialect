use melior::{
    Context,
    dialect::{func, DialectRegistry},
    ExecutionEngine,
    ir::{
        attribute::{StringAttribute, TypeAttribute},
        Attribute,
        r#type::{FunctionType, IntegerType},
        Block, BlockLike, Location, Module, Operation, Region, RegionLike,
    },
    pass::{self, PassManager},
    utility::{register_all_dialects},
};
use str_dialect as str_;

fn build_test1_func<'c>(
    context: &'c Context,
    loc: Location<'c>
) -> Operation<'c> {
    // build a func.func @test1:
    //
    // func.func @test1() -> i1 {
    //   %a = str.constant "hello" : !str.string
    //   %b = str.constant "hello" : !str.string
    //   %r = str.cmp eq, %a, %b : !str.string
    //   return %r : i1
    // }

    // build the function body
    let region = {
        let block = Block::new(&[]);

        let a = block.append_operation(str_::constant(
            loc,
            "hello",
        )).result(0).unwrap().into();

        let b = block.append_operation(str_::constant(
            loc,
            "hello",
        )).result(0).unwrap().into();

        let r = block.append_operation(str_::cmp(
            loc,
            str_::CmpPredicate::Eq,
            a,
            b,
        )).result(0).unwrap().into();

        block.append_operation(func::r#return(
            &[r],
            loc,
        ));

        let region = Region::new();
        region.append_block(block);
        region
    };

    // build the function
    let function_type = FunctionType::new(
        &context,
        &[],
        &[IntegerType::new(&context, 1).into()]
    );

    let mut func_op = func::func(
        &context,
        StringAttribute::new(&context, "test1"),
        TypeAttribute::new(function_type.into()),
        region,
        &[],
        loc,
    );
    func_op.set_attribute("llvm.emit_c_interface", Attribute::unit(&context));
    func_op
}

fn parse_operation_from_string<'c>(
    context: &'c Context,
    source: &str,
) -> Option<Operation<'c>> {
    // Convert the source string to a C-compatible string
    let c_source = std::ffi::CString::new(source).ok()?;
    let c_source_ref = unsafe { mlir_sys::mlirStringRefCreateFromCString(c_source.as_ptr()) };

    // Convert the location to a string for the source name
    let c_source_name = std::ffi::CString::new("source").ok()?;
    let c_source_name_ref = unsafe { mlir_sys::mlirStringRefCreateFromCString(c_source_name.as_ptr()) };

    // Parse the operation
    let operation = unsafe {
        mlir_sys::mlirOperationCreateParse(context.to_raw(), c_source_ref, c_source_name_ref)
    };

    if operation.ptr.is_null() {
        None
    } else {
        Some(unsafe { Operation::from_raw(operation) })
    }
}

fn build_test2_func<'c>(context: &'c Context) -> Operation<'c> {
    // this function applies str.cmp on a & b
    // for each different predicate
    // and packs the result of each application into
    // a bit vector
    let source = r#"
    func.func @test2(%a: !str.string, %b: !str.string) -> i64 {
      %eq = str.cmp eq, %a, %b : !str.string
      %ne = str.cmp ne, %a, %b : !str.string
      %lt = str.cmp lt, %a, %b : !str.string
      %le = str.cmp le, %a, %b : !str.string
      %gt = str.cmp gt, %a, %b : !str.string
      %ge = str.cmp ge, %a, %b : !str.string
    
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c3 = arith.constant 3 : i64
      %c4 = arith.constant 4 : i64
      %c5 = arith.constant 5 : i64
    
      %eq64 = arith.extui %eq : i1 to i64
      %ne64 = arith.extui %ne : i1 to i64
      %lt64 = arith.extui %lt : i1 to i64
      %le64 = arith.extui %le : i1 to i64
      %gt64 = arith.extui %gt : i1 to i64
      %ge64 = arith.extui %ge : i1 to i64
    
      %ne_shifted = arith.shli %ne64, %c1 : i64
      %lt_shifted = arith.shli %lt64, %c2 : i64
      %le_shifted = arith.shli %le64, %c3 : i64
      %gt_shifted = arith.shli %gt64, %c4 : i64
      %ge_shifted = arith.shli %ge64, %c5 : i64
    
      %result1 = arith.ori %eq64, %ne_shifted : i64
      %result2 = arith.ori %result1, %lt_shifted : i64
      %result3 = arith.ori %result2, %le_shifted : i64
      %result4 = arith.ori %result3, %gt_shifted : i64
      %bitmask = arith.ori %result4, %ge_shifted : i64
    
      return %bitmask : i64
    }
    "#;

    let mut op = parse_operation_from_string(
        context,
        source,
    ).unwrap();

    op.set_attribute("llvm.emit_c_interface", Attribute::unit(&context));
    op
}

#[repr(C)]
#[derive(Debug)]
struct MemRef1D {
    allocated: *const u8,
    aligned: *const u8,
    offset: i64,
    size: i64,
    stride: i64,
}

/// Create a memref<?xi8> descriptor from a slice
fn make_memref_descriptor(slice: &[u8]) -> MemRef1D {
    let ptr = slice.as_ptr();
    let len = slice.len() as i64;

    MemRef1D {
        allocated: ptr,
        aligned: ptr,
        offset: 0,
        size: len,
        stride: 1,
    }
}

#[test]
fn test_str_jit() {
    // create a dialect registry and register all dialects
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    str_::register(&context);

    // make all the dialects available
    context.load_all_available_dialects();

    // create a module
    let loc = Location::unknown(&context);
    let mut module = Module::new(loc);

    // build two functions @test1 and @test2
    module.body().append_operation(
        build_test1_func(&context, loc)
    );
    module.body().append_operation(
        build_test2_func(&context)
    );

    assert!(module.as_operation().verify(), "MLIR module verification failed");

    // Lower to LLVM
    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    assert!(pass_manager.run(&mut module).is_ok());

    // JIT compile the module
    let engine = ExecutionEngine::new(&module, 0, &[], false);

    // test 1
    unsafe {
        let mut result: bool = false;
        let mut packed_args: [*mut (); 1] = [
            &mut result as *mut bool as *mut ()
        ];

        engine.invoke_packed("test1", &mut packed_args)
            .expect("test1 JIT invocation failed");

        assert_eq!(result, true);
    }

    // test 2

    // this helper function helps us actually call the test2 function
    fn call_test2(engine: &ExecutionEngine, a: &[u8], b: &[u8]) -> i64 {
        // each input must be null-terminated
        assert!(a.ends_with(&[0]) && b.ends_with(&[0]));

        // Prepare arguments (as `char*`, i.e., `*mut i8`)
        let mut memref_a = make_memref_descriptor(a);
        let mut memref_b = make_memref_descriptor(b);
        let mut result: i64 = -1;

        let mut packed_args: [*mut (); 3] = [
            &mut memref_a as *mut _ as *mut (),
            &mut memref_b as *mut _ as *mut (),
            &mut result as *mut _ as *mut (),
        ];

        unsafe {
            engine
                .invoke_packed("test2", &mut packed_args)
                .expect("test2 invocation failed");
        }

        result
    }

    // Bit layout of str.cmp result (least significant bit on the right):
    //
    //   [ ge gt le lt ne eq ]
    //     5  4  3  2  1  0
    //
    // Each bit is set to 1 if the corresponding predicate evaluates to true.

    // true bits: eq, le, ge 
    assert_eq!(call_test2(&engine, b"abc\0", b"abc\0"), 0b101001);
    
    // true bits: ne, lt, le
    assert_eq!(call_test2(&engine, b"a\0", b"abc\0"), 0b001110);
    
    // true bits: ne, gt, ge
    assert_eq!(call_test2(&engine, b"abc\0", b"a\0"), 0b110010);
    
    // true bits: ne, lt, le
    assert_eq!(call_test2(&engine, b"abc\0", b"abcd\0"), 0b001110);
    
    // true bits: ne, gt, ge
    assert_eq!(call_test2(&engine, b"abcd\0", b"abc\0"), 0b110010);
    
    // true bits: eq, le, ge
    assert_eq!(call_test2(&engine, b"\0", b"\0"), 0b101001);
}
