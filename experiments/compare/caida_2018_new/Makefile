# 编译器设置
CXX = g++
CXXFLAGS = -std=c++11 -Wall -O3 -mavx2 -mbmi -mbmi2 -Wno-psabi -Wno-sign-compare -Wno-unused-variable

# 默认参数设置

ELASTIC_TOT_MEM ?= 409600
ELASTIC_HEAVY_MEM ?= 102400
MRAC_TOT_MEM ?= 409600
SAMPLE_RATE ?= 0.1

# 添加参数到编译标志
CXXFLAGS += -DELASTIC_HEAVY_MEM=$(ELASTIC_HEAVY_MEM) \
            -DELASTIC_TOT_MEM=$(ELASTIC_TOT_MEM) \
            -DMRAC_TOT_MEM=$(MRAC_TOT_MEM) \
            -DSAMPLE_RATE=$(SAMPLE_RATE) \
            -DMAX_PRIME32=1229

# 目录设置
SKETCH_DIR = src/Sketchs
COMMON_DIR = src/common
BUILD_DIR = build
BIN_DIR = bin

# 头文件路径
INCLUDES = -I. -I$(SKETCH_DIR) -I$(COMMON_DIR)

# 源文件
SRCS = src/main.cpp
COMMON_SRCS = $(wildcard $(COMMON_DIR)/*.cpp)
SKETCH_SRCS = $(wildcard $(SKETCH_DIR)/elastic/*.cpp) $(wildcard $(SKETCH_DIR)/MRAC/*.cpp)

# 目标文件
OBJS = $(SRCS:%.cpp=$(BUILD_DIR)/%.o)
COMMON_OBJS = $(COMMON_SRCS:%.cpp=$(BUILD_DIR)/%.o)
SKETCH_OBJS = $(SKETCH_SRCS:%.cpp=$(BUILD_DIR)/%.o)

# 依赖文件
DEPS = $(OBJS:.o=.d) $(COMMON_OBJS:.o=.d) $(SKETCH_OBJS:.o=.d)

# 最终可执行文件
TARGET = $(BIN_DIR)/sketch_test

# 默认目标
all: directories $(TARGET)

# 创建必要的目录
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)/$(COMMON_DIR)
	@mkdir -p $(BUILD_DIR)/$(SKETCH_DIR)/elastic
	@mkdir -p $(BUILD_DIR)/$(SKETCH_DIR)/MRAC
	@mkdir -p $(BIN_DIR)

# 链接规则
$(TARGET): $(OBJS) $(COMMON_OBJS) $(SKETCH_OBJS)
	@echo "Linking $@..."
	@$(CXX) $(CXXFLAGS) $^ -o $@
	@echo "Build complete!"

# 编译规则，生成依赖文件
$(BUILD_DIR)/%.o: %.cpp
	@echo "Compiling $<..."
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# 编译common目录下的源文件
$(BUILD_DIR)/$(COMMON_DIR)/%.o: $(COMMON_DIR)/%.cpp
	@echo "Compiling $<..."
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# 编译Sketch目录下的源文件
$(BUILD_DIR)/$(SKETCH_DIR)/%.o: $(SKETCH_DIR)/%.cpp
	@echo "Compiling $<..."
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# 清理规则
clean:
	@echo "Cleaning build files..."
	@rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "Clean complete!"

# 帮助信息
help:
	@echo "Available targets:"
	@echo "  all          - Build the project (default)"
	@echo "  clean        - Remove build files"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Configuration options:"
	@echo "  ELASTIC_HEAVY_MEM=<bytes>  - Set Elastic sketch heavy part memory (default: 153600)"
	@echo "  ELASTIC_TOT_MEM=<bytes>    - Set Elastic sketch total memory (default: 614400)"
	@echo "  MRAC_TOT_MEM=<bytes>       - Set MRAC total memory (default: 102400)"
	@echo "  SAMPLE_RATE=<rate>         - Set MRAC sample rate (default: 0.1)"
	@echo ""
	@echo "Example:"
	@echo "  make ELASTIC_HEAVY_MEM=307200 ELASTIC_TOT_MEM=1228800"

# 伪目标声明
.PHONY: all clean help directories

# 包含依赖文件
-include $(DEPS)
