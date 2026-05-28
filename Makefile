# QuantForge - 超低延迟 AI 量化交易系统
# 统一自动化构建脚本

LIBTORCH_DIR = /data/libtorch
CXX = g++
CXXFLAGS = -std=c++17 -O3 -fPIC -shared
INCLUDES = -I$(LIBTORCH_DIR)/include -I$(LIBTORCH_DIR)/include/torch/csrc/api/include
LDFLAGS = -L$(LIBTORCH_DIR)/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$(LIBTORCH_DIR)/lib

.PHONY: all ocaml c_lib scheme clean test

# 默认目标
all: c_lib ocaml scheme

# 构建 OCaml 模块
ocaml:
	@echo "Building OCaml 4 ingestion engine..."
	opam exec -- dune build
	@echo "OCaml build complete."

# 构建 C++ 底层库
c_lib:
	@echo "Building C++ core library with LibTorch..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	  -o c_src/libquant_core.so c_src/libquant_core.cpp \
	  $(LDFLAGS)
	@echo "C++ build complete: c_src/libquant_core.so"

# 构建 Chez Scheme 策略模块
scheme:
	@echo "Chez Scheme modules ready (scheme_src/)"

# 运行测试
test: ocaml
	@echo "Running tests..."
	opam exec -- dune test

# 清理构建产物
clean:
	@echo "Cleaning build artifacts..."
	opam exec -- dune clean
	rm -f c_src/*.so c_src/*.o
	@echo "Clean complete."

# 安装依赖
deps:
	@echo "Installing OCaml dependencies..."
	opam install -y lwt lwt_ppx

# 运行行情接入引擎
run: ocaml
	@echo "Starting QuantForge ingestion engine..."
	opam exec -- dune exec quantforge-ingestion

# 运行策略引擎
run-scheme:
	/data/ChezScheme/pb/bin/pb/petite --script scheme_src/main.ss

# 帮助信息
help:
	@echo "QuantForge Build System"
	@echo "======================="
	@echo "  all      - Build all components"
	@echo "  ocaml    - Build OCaml 4 ingestion engine"
	@echo "  c_lib    - Build C++ LibTorch library"
	@echo "  scheme   - Ready Chez Scheme modules"
	@echo "  test     - Run tests"
	@echo "  clean    - Clean build artifacts"
	@echo "  run      - Run ingestion engine"
	@echo "  run-scheme - Run strategy engine"
