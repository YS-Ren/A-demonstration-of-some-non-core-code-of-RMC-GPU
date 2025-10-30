#ifndef __GPU_PRINT_H__
#define  __GPU_PRINT_H__

#include<stdio.h>
#include<iostream>
#include<sstream>
#include<string>
#include<vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// wingdi.h头文件定义了ERROR宏，与CDLogger::eLevel::ERROR冲突
#undef ERROR

//**************************************************
//	class def
//**************************************************

// 前景色
enum class eColorType {
    BLACK = 30,
    RED = 31,
    GREEN = 32,
    YELLOW = 33,
    BLUE = 34,
    MAGENTA = 35,
    CYAN = 36,
    WHITE = 37,
    DEFAULT = 39,
    BRIGHT_BLACK = 90,
    BRIGHT_RED = 91,
    BRIGHT_GREEN = 92,
    BRIGHT_YELLOW = 93,
    BRIGHT_BLUE = 94,
    BRIGHT_MAGENTA = 95,
    BRIGHT_CYAN = 96,
    BRIGHT_WHITE = 97
};

// 背景色
enum class eBackgroundType {
    BLACK = 40,
    RED = 41,
    GREEN = 42,
    YELLOW = 43,
    BLUE = 44,
    MAGENTA = 45,
    CYAN = 46,
    WHITE = 47,
    DEFAULT = 49,
    BRIGHT_BLACK = 100,
    BRIGHT_RED = 101,
    BRIGHT_GREEN = 102,
    BRIGHT_YELLOW = 103,
    BRIGHT_BLUE = 104,
    BRIGHT_MAGENTA = 105,
    BRIGHT_CYAN = 106,
    BRIGHT_WHITE = 107
};

// 文本样式
enum class eStyleType {
    RESET = 0,
    BOLD = 1,
    DIM = 2,
    ITALIC = 3,
    UNDERLINE = 4,
    BLINK = 5,
    REVERSE = 7,
    HIDDEN = 8
};

// 文本对齐方式
enum class eAlignType {
    LEFT,
    RIGHT,
    CENTER
};

// 颜色控制
class CDColorizer {
public:
    std::string set(eColorType color);

    std::string set(eBackgroundType background);

    std::string set(eStyleType style);

    std::string reset();
};

extern CDColorizer OColorizer;

#define OUTPUT_COLOR(color) OColorizer.set(eColorType::color)
#define OUTPUT_BG(color) OColorizer.set(eBackgroundType::color)
#define OUTPUT_STYLE(style) OColorizer.set(eStyleType::style)
#define RESET OColorizer.reset()

// 格式化输出
class CDFormatter {
public:
    // 设置文本对齐
    std::string align(std::string text, int width, eAlignType alignment = eAlignType::LEFT, char fill = ' ');
    // 创建进度条
    std::string progressBar(float progress, int width = 20);
};

extern CDFormatter OFormatter;

// 表格输出
class CDTable {
public:
    CDTable() : showBorder(true), showHeader(true) {}

    // 添加列
    void addColumn(const std::string& name, eAlignType alignment = eAlignType::LEFT);

    // 添加行
    void addRow(const std::vector<std::string>& row);

    // 设置是否显示边框
    void setShowBorder(bool show);

    // 设置是否显示表头
    void setShowHeader(bool show);

    // 输出表格
    void print(std::ostream& os = std::cout);

private:
    struct CDColumn {
        std::string name;
        eAlignType alignment;
    };

    std::vector<CDColumn> columns;
    std::vector<std::vector<std::string>> rows;
    std::vector<size_t> colWidths;
    bool showBorder;
    bool showHeader;

    void printHorizontalBorder(std::ostream& os);

    void printHorizontalSeparator(std::ostream& os);

    void printRow(std::ostream& os, std::vector<std::string>& row, bool isHeader);

    std::vector<std::string> getHeaderRow();
};

// 日志输出
class CDLogger {
public:
    enum eLevel {
        DEBUG,
        INFO,
        WARNING,
        ERROR,
        CRITICAL
    };

    void setLevel(eLevel level);

    void debug(const std::string& message, const std::string& component = "");

    void info(const std::string& message, const std::string& component = "");

    void warning(const std::string& message, const std::string& component = "");

    void error(const std::string& message, const std::string& component = "");

    void critical(const std::string& message, const std::string& component = "");

    void header(const std::string& message, const std::string& component = "", const eLevel level = eLevel::INFO);

    void main(const std::string& message, const std::string& component = "");

private:
    eLevel logLevel{ CDLogger::DEBUG };

    void log(eLevel level, const std::string& message, const std::string& component);
};

__host__ __device__ const char* GetFilename_GPUd(const char* path);

/**
 * @brief unexpected branch % only used in [device]
 */
__device__ void unexpectedBranch(char* file, int line);

__device__ void BreakPoint_GPUd(char* file, int line);

//**************************************************
//	extern var
//**************************************************

extern CDLogger OLogger;

// 返回不包含路径的文件名 host and device
#define __FILE_GPU__ GetFilename_GPUd(__FILE__)
// host only!
#define __POSITION___ " at " << __FILE_GPU__ << ":" << __LINE__
// host only!
#define __POSITION_ENDL__ " at " << __FILE_GPU__ << ":" << __LINE__ << std::endl

#define UnexpectedBranch unexpectedBranch(__FILE_GPU__, __LINE__);

#define BreakPoint_GPU(tid) \
    if(blockIdx.x * blockDim.x + threadIdx.x==tid){ \
		BreakPoint_GPUd(__FILE_GPU__, __LINE__); \
	}


#endif //__GPU_PRINT_H__