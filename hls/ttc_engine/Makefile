# C-simulation with g++ and the Vitis HLS headers (no device library needed):
#   make csim XILINX_HLS_INCLUDE=/mnt/c/Xilinx/Vitis/2024.2/include
# Synthesis (needs Zynq UltraScale+ device data installed for Vitis):
#   vitis-run --mode hls --csynth --config hls_config.cfg --work_dir build
XILINX_HLS_INCLUDE ?= /mnt/c/Xilinx/Vitis/2024.2/include
CXXFLAGS = -std=c++17 -O2 -Wno-unknown-pragmas -I$(XILINX_HLS_INCLUDE) -DGOLDEN_DIR=\"golden\"

csim: build/csim
	./build/csim

build/csim: src/ttc_engine.cpp src/ttc_engine.hpp tb/tb_ttc_engine.cpp
	mkdir -p build
	g++ $(CXXFLAGS) src/ttc_engine.cpp tb/tb_ttc_engine.cpp -o $@

.PHONY: csim
