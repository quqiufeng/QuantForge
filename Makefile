# QuantForge Makefile

LIBTORCH_DIR = /data/libtorch_cuda
CUDA_DIR = /data/cuda

CXX = g++
CXXFLAGS = -std=c++17 -O3 -fPIC -shared -DUSE_CUDA
INCLUDES = -I$(LIBTORCH_DIR)/include -I$(LIBTORCH_DIR)/include/torch/csrc/api/include -I$(CUDA_DIR)/include
LDFLAGS = -L$(LIBTORCH_DIR)/lib -L$(CUDA_DIR)/lib64 -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lcudart -Wl,-rpath,$(LIBTORCH_DIR)/lib -Wl,-rpath,$(CUDA_DIR)/lib64

CC = gcc
CFLAGS = -fPIC -O2 -I$(LIBTORCH_DIR)/include
OCAML_CFLAGS = $(shell opam config var ctypes:lib)/../include

.PHONY: all c_lib ocaml clean run help

all: c_lib ocaml

c_lib:
	@echo "[1/2] Building C++ CUDA library..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o c_src/libquant_core.so c_src/libquant_core.cpp $(LDFLAGS)
	@echo "      Building OCaml bridge..."
	$(CC) $(CFLAGS) -I/usr/lib/ocaml -c c_src/ocaml_bridge.c -o c_src/ocaml_bridge.o
	@echo "      Linking bridge..."
	$(CC) -shared -o c_src/libquant_bridge.so c_src/ocaml_bridge.o -Lc_src -lquant_core -Wl,-rpath,c_src
	@echo "      C++ build complete"

ocaml: c_lib
	@echo "[2/2] Building OCaml..."
	LD_LIBRARY_PATH=c_src:$(LIBTORCH_DIR)/lib:$(CUDA_DIR)/lib64:$$LD_LIBRARY_PATH \
	opam exec -- dune build
	@echo "      OCaml build complete"

clean:
	opam exec -- dune clean
	rm -f c_src/*.so c_src/*.o

run: all
	@echo "Starting QuantForge..."
	LD_LIBRARY_PATH=c_src:$(LIBTORCH_DIR)/lib:$(CUDA_DIR)/lib64:$$LD_LIBRARY_PATH \
	opam exec -- dune exec quantforge-ingestion

help:
	@echo "Targets: all, c_lib, ocaml, clean, run"
