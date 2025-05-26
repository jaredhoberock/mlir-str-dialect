LLVM_BIN_PATH = $(HOME)/dev/git/llvm-project-20/build/bin

LLVM_CONFIG := $(LLVM_BIN_PATH)/llvm-config
TBLGEN := $(LLVM_BIN_PATH)/mlir-tblgen
OPT := $(LLVM_BIN_PATH)/mlir-opt

# Compiler flags
CXX := clang++
CXXFLAGS := -g -fPIC `$(LLVM_CONFIG) --cxxflags`

# LLVM/MLIR libraries
MLIR_INCLUDE = $(HOME)/dev/git/llvm-project-20/install/include

INCLUDES := -I $(MLIR_INCLUDE)

# Dialect library sources
DIALECT_SOURCES := c_api.cpp Dialect.cpp Enums.cpp Lowering.cpp Ops.cpp Types.cpp
DIALECT_OBJECTS := $(DIALECT_SOURCES:.cpp=.o)

# Generated files
GENERATED := Dialect.hpp.inc Dialect.cpp.inc Enums.hpp.inc Enums.cpp.inc Ops.hpp.inc Ops.cpp.inc Types.hpp.inc Types.cpp.inc

.PHONY: all clean

all: libstr_dialect.a libstr_dialect.so

# TableGen rules
Dialect.hpp.inc: Dialect.td
	$(TBLGEN) --gen-dialect-decls $(INCLUDES) $< -o $@

Dialect.cpp.inc: Dialect.td
	$(TBLGEN) --gen-dialect-defs $(INCLUDES) $< -o $@

Enums.hpp.inc: Enums.td
	$(TBLGEN) --gen-enum-decls $(INCLUDES) $< -o $@

Enums.cpp.inc: Enums.td
	$(TBLGEN) --gen-enum-defs $(INCLUDES) $< -o $@

Ops.hpp.inc: Ops.td
	$(TBLGEN) --gen-op-decls $(INCLUDES) $< -o $@

Ops.cpp.inc: Ops.td
	$(TBLGEN) --gen-op-defs $(INCLUDES) $< -o $@

Types.hpp.inc: Types.td
	$(TBLGEN) --gen-typedef-decls $(INCLUDES) $< -o $@

Types.cpp.inc: Types.td
	$(TBLGEN) --gen-typedef-defs $(INCLUDES) $< -o $@

# Object file rules
%.o: %.cpp $(GENERATED)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

libstr_dialect.a: $(DIALECT_OBJECTS)
	ar rcs $@ $^

PLUGIN_OBJECTS := $(DIALECT_OBJECTS) Plugin.o

libstr_dialect.so: $(PLUGIN_OBJECTS)
	$(CXX) -shared $^ -o $@

.PHONY: test
test: libstr_dialect.so
	@echo "Running str dialect tests..."
	lit tests

clean:
	rm -f *.o *.inc *.a *.so
