# Build all binaries with the same flags. CXX_MARCH defaults to "haswell"
# to match the rinha reference Mac mini 2014 (i5-4278U). For local dev or
# bench on other machines, override with `CXX_MARCH=native` or `x86-64`.
CXX      ?= g++
CXX_MARCH ?= haswell
CXXFLAGS ?= -O3 -std=c++20 -march=$(CXX_MARCH) -mtune=$(CXX_MARCH) \
            -fno-exceptions -fno-rtti -fno-plt -fvisibility=hidden \
            -ffunction-sections -fdata-sections \
            -Wall -Wextra -Wno-unused-parameter
LDFLAGS  ?= -Wl,--gc-sections -static-libstdc++ -static-libgcc

BIN_DIR  ?= build

.PHONY: all clean api lb

all: $(BIN_DIR)/api $(BIN_DIR)/lb $(BIN_DIR)/eval $(BIN_DIR)/k_means

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/api: src/http/server.cpp src/engine/engine.hpp src/engine/ivf_index.hpp src/engine/parser.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(BIN_DIR)/lb: src/lb/lb.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(BIN_DIR)/eval: src/tools/eval.cpp src/engine/engine.hpp src/engine/ivf_index.hpp src/engine/parser.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Training builds with -march=native (offline, runs on the build host).
$(BIN_DIR)/k_means: src/training/k_means.cpp src/engine/ivf_index.hpp | $(BIN_DIR)
	$(CXX) -O3 -std=c++20 -march=native -mtune=native $(if $(filter $(CXX),g++ c++),-Wall,) $< -o $@

api: $(BIN_DIR)/api
lb:  $(BIN_DIR)/lb

clean:
	rm -rf $(BIN_DIR)
