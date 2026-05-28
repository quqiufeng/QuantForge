# QuantForge - 超低延迟 AI 量化交易系统 (RTX 3080 20G CUDA 版)

# 路径配置
LIBTORCH_DIR = /data/libtorch_cuda
CUDA_DIR = /data/cuda

# 编译器
CXX = g++
NVCC = $(CUDA_DIR)/bin/nvcc

# 编译选项
CXXFLAGS = -std=c++17 -O3 -fPIC -shared -DUSE_CUDA
CUDAFLAGS = -arch=sm_86  # RTX 3080 = Ampere = sm_86

# 头文件路径
INCLUDES = -I$(LIBTORCH_DIR)/include \
           -I$(LIBTORCH_DIR)/include/torch/csrc/api/include \
           -I$(CUDA_DIR)/include

# 库文件路径和链接
LDFLAGS = -L$(LIBTORCH_DIR)/lib \
          -L$(CUDA_DIR)/lib64 \
          -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
          -lcudart -lcublas -lcudnn \
          -Wl,-rpath,$(LIBTORCH_DIR)/lib \
          -Wl,-rpath,$(CUDA_DIR)/lib64

.PHONY: all ocaml c_lib scheme clean test run help

all: c_lib ocaml scheme

# OCaml 4 行情接入引擎
ocaml:
	@echo "[1/3] Building OCaml 4 ingestion engine..."
	opam exec -- dune build
	@echo "      OCaml build complete."

# C++ CUDA 推理库
c_lib:
	@echo "[2/3] Building C++ CUDA library for RTX 3080..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	  -o c_src/libquant_core.so c_src/libquant_core.cpp \
	  $(LDFLAGS)
	@echo "      C++ CUDA build complete: c_src/libquant_core.so"

# Chez Scheme 策略引擎
scheme:
	@echo "[3/3] Chez Scheme modules ready (scheme_src/)"

# 运行测试
test: ocaml
	@echo "Running tests..."
	opam exec -- dune test

# 清理
clean:
	@echo "Cleaning build artifacts..."
	opam exec -- dune clean
	rm -f c_src/*.so c_src/*.o

# 安装依赖
deps:
	opam install -y lwt lwt_ppx

# 运行行情引擎
run: ocaml
	@echo "Starting QuantForge (RTX 3080 CUDA mode)..."
	opam exec -- dune exec quantforge-ingestion

# 运行策略引擎
run-scheme:
	$(SCHEME_BIN) --script scheme_src/main.ss

# 验证 CUDA 环境
verify-cuda:
	@echo "=== CUDA Environment ==="
	@nvcc --version
	@echo ""
	@echo "=== GPU Status ==="
	@nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
	@echo ""
	@echo "=== LibTorch CUDA ==="
	@ls $(LIBTORCH_DIR)/lib/libtorch_cuda.so 2>/dev/null && echo "OK" || echo "NOT FOUND"

help:
	@echo "QuantForge Build System (RTX 3080 20G)"
	@echo "======================================="
	@echo "  all         - Build all components"
	@echo "  ocaml       - Build OCaml ingestion engine"
	@echo "  c_lib       - Build C++ CUDA library"
	@echo "  scheme      - Chez Scheme modules"
	@echo "  test        - Run tests"
	@echo "  clean       - Clean build artifacts"
	@echo "  run         - Run ingestion engine"
	@echo "  verify-cuda - Verify CUDA environment"
