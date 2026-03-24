CXX = clang++
CC = clang
INCLUDES = -I. -I./include -I./src -I./ggml/include -I./ggml/src -I./ggml/src/ggml-metal -I./ggml/src/ggml-cpu -O3 -DGGML_USE_METAL -DGGML_USE_CPU -DNDEBUG -DGGML_VERSION=\"\" -DGGML_COMMIT=\"\" -DGGML_CPU_ARM_NEON -DGGML_CPU_ARM_FEAT_DOTPROD
CFLAGS = $(INCLUDES) -fno-objc-arc
CXXFLAGS = $(CFLAGS) -std=c++17
LDFLAGS = -framework Foundation -framework Metal -framework MetalKit

SRCS_CPP = main.cpp \
           kv_zlib.cpp \
           $(wildcard src/*.cpp) \
           $(wildcard src/models/*.cpp) \
           ggml/src/ggml.cpp \
           ggml/src/ggml-backend.cpp \
           ggml/src/ggml-backend-reg.cpp \
           ggml/src/ggml-opt.cpp \
           ggml/src/ggml-threading.cpp \
           ggml/src/gguf.cpp \
           ggml/src/ggml-metal/ggml-metal.cpp \
           ggml/src/ggml-metal/ggml-metal-ops.cpp \
           ggml/src/ggml-metal/ggml-metal-device.cpp \
           ggml/src/ggml-metal/ggml-metal-common.cpp \
           ggml/src/ggml-cpu/arch/arm/repack.cpp \
           ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp \
           $(wildcard ggml/src/ggml-cpu/*.cpp)

SRCS_C = ggml/src/ggml.c \
         ggml/src/ggml-alloc.c \
         ggml/src/ggml-quants.c \
         ggml/src/ggml-cpu/arch/arm/quants.c \
         $(wildcard ggml/src/ggml-cpu/*.c)

SRCS_M = ggml/src/ggml-metal/ggml-metal-device.m \
         ggml/src/ggml-metal/ggml-metal-context.m

OBJS = $(SRCS_CPP:.cpp=.cpp.o) $(SRCS_C:.c=.c.o) $(SRCS_M:.m=.m.o)

gemma_chat: $(OBJS)
	$(CXX) $(OBJS) -o gemma_chat $(LDFLAGS) -lz

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.m.o: %.m
	$(CC) $(CFLAGS) -fno-objc-arc -c $< -o $@

clean:
	rm -f gemma_chat $(OBJS)
