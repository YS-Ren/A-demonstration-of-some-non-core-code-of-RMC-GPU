#include"GPUPrint.h"

#include"GPUBaseFunc.h"

//**************************************************
//	class def
//**************************************************

// 颜色控制
// class CDColorizer 
CDColorizer OColorizer;
std::string CDColorizer::set(eColorType color) {
    return "\033[" + std::to_string(static_cast<int>(color)) + "m";
}

std::string CDColorizer::set(eBackgroundType background) {
    return "\033[" + std::to_string(static_cast<int>(background)) + "m";
}

std::string CDColorizer::set(eStyleType style) {
    return "\033[" + std::to_string(static_cast<int>(style)) + "m";
}

std::string CDColorizer::reset() {
    return "\033[0m";
}


// 格式化输出
// class CDFormatter 
CDFormatter OFormatter;
std::string CDFormatter::align(std::string text, int width, eAlignType alignment, char fill) {
    if (width <= static_cast<int>(text.length())) {
        return text;
    }

    switch (alignment) {
    case eAlignType::LEFT:
        return text + std::string(width - text.length(), fill);
    case eAlignType::RIGHT:
        return std::string(width - text.length(), fill) + text;
    case eAlignType::CENTER: {
        int left = (width - text.length()) / 2;
        int right = width - text.length() - left;
        return std::string(left, fill) + text + std::string(right, fill);
    }
    default:
        return text;
    }
}

// 进度条
std::string CDFormatter::progressBar(float progress, int width) {
    int pos = static_cast<int>(width * progress);
    std::string bar;

    for (int i = 0; i < width; ++i) {
        if (i < pos) bar += "=";
        else if (i == pos) bar += ">";
        else bar += " ";
    }

    return "[" + bar + "] " + std::to_string(static_cast<int>(progress * 100)) + "%";
}

// 表格输出
//class CDTable 
void CDTable::addColumn(const std::string& name, eAlignType alignment) {
    columns.push_back({ name, alignment });
    colWidths.push_back(name.length());
}

void CDTable::addRow(const std::vector<std::string>& row) {
    if (row.size() != columns.size()) {
        throw std::runtime_error("Row size doesn't match number of columns");
    }

    rows.push_back(row);

    // 更新列宽
    for (size_t i = 0; i < row.size(); ++i) {
        colWidths[i] = std::max(colWidths[i], row[i].length());
    }
}

void CDTable::setShowBorder(bool show) {
    showBorder = show;
}

void CDTable::setShowHeader(bool show) {
    showHeader = show;
}

void CDTable::print(std::ostream& os) {
    if (columns.empty()) return;

    // 打印上边框
    if (showBorder) {
        printHorizontalBorder(os);
    }

    // 打印表头
    if (showHeader) {
        printRow(os, getHeaderRow(), true);
        if (showBorder) {
            printHorizontalBorder(os);
        }
    }

    // 打印数据行
    for (size_t i = 0; i < rows.size(); ++i) {
        printRow(os, rows[i], false);
        if (showBorder && i < rows.size() - 1) {
            printHorizontalSeparator(os);
        }
    }

    // 打印下边框
    if (showBorder) {
        printHorizontalBorder(os);
    }
}

void CDTable::printHorizontalBorder(std::ostream& os) {
    os << "\t+";
    for (size_t width : colWidths) {
        os << std::string(width + 2, '-') << "+";
    }
    os << std::endl;
}

void CDTable::printHorizontalSeparator(std::ostream& os) {
    os << "\t|";
    for (size_t width : colWidths) {
        os << std::string(width + 2, '-') << "+";
    }
    os << std::endl;
}

void CDTable::printRow(std::ostream& os, std::vector<std::string>& row, bool isHeader) {
    os << "\t|";
    for (size_t i = 0; i < row.size(); ++i) {
        std::string cell = OFormatter.align(row[i], colWidths[i], columns[i].alignment);
        if (isHeader) {
            cell = OUTPUT_STYLE(BOLD) + cell + RESET;
        }
        os << " " << cell << " |";
    }
    os << std::endl;
}

std::vector<std::string> CDTable::getHeaderRow() {
    std::vector<std::string> header;
    for (auto& col : columns) {
        header.push_back(col.name);
    }
    return header;
}

// 日志输出
// class CDLogger {
void CDLogger::setLevel(eLevel level) {
    logLevel = level;
}

void CDLogger::debug(const std::string& message, const std::string& component) {
    log(DEBUG, message, component);
}

void CDLogger::info(const std::string& message, const std::string& component) {
    log(INFO, message, component);
}

void CDLogger::warning(const std::string& message, const std::string& component) {
    log(WARNING, message, component);
}

void CDLogger::error(const std::string& message, const std::string& component) {
    log(ERROR, message, component);
}

void CDLogger::critical(const std::string& message, const std::string& component) {
    log(CRITICAL, message, component);
}

void CDLogger::header(const std::string& message, const std::string& component, const eLevel level) {
    switch (level) {
    case DEBUG:
        debug(message, component);
        break;
    case INFO:
        info(message, component);
        break;
    case WARNING:
        warning(message, component);
        break;
    case ERROR:
        error(message, component);
        break;
    case CRITICAL:
        critical(message, component);
        break;
    }
}

void CDLogger::main(const std::string& message, const std::string& component) {
    std::string compStr = component.empty() ? "" : "[" + component + "] ";
    std::cout << "\t -> " << RESET << compStr << message << std::endl;
}

void CDLogger::log(eLevel level, const std::string& message, const std::string& component) {
    if (level < logLevel) return;

    std::string prefix;

    switch (level) {
    case DEBUG:
        prefix = "DEBUG";
        std::cout << OUTPUT_COLOR(BRIGHT_BLACK);
        break;
    case INFO:
        prefix = "INFO";
        std::cout << OUTPUT_COLOR(BRIGHT_BLUE);
        break;
    case WARNING:
        prefix = "WARN";
        std::cout << OUTPUT_COLOR(BRIGHT_YELLOW);
        break;
    case ERROR:
        prefix = "ERROR";
        std::cout << OUTPUT_COLOR(BRIGHT_RED);
        break;
    case CRITICAL:
        prefix = "CRIT";
        std::cout << OUTPUT_COLOR(RED);
        break;
    }

    std::string compStr = component.empty() ? "" : "[" + component + "] ";
    std::cout << "[" << prefix << "]\t" << RESET << compStr << message << std::endl;
}

__host__ __device__ const char* GetFilename_GPUd(const char* path) {
    const char* filename = strrchr_GPUd(path, '/');
    if (filename == nullptr) {
        filename = strrchr_GPUd(path, '\\');
    }
    return (filename != nullptr) ? filename + 1 : path;
}

__device__ void unexpectedBranch(char* file, int line) {
    printf("Warning: Program run into an unexpected branch! from block[%d] thread[%d] at file:\"%s\" | line:%d\n", blockIdx.x, threadIdx.x, file, line);
}

__device__ void BreakPoint_GPUd(char* file, int line) {
    printf("BreakPoint from block[%d] thread[%d] at file:\"%s\" | line:%d\n", blockIdx.x, threadIdx.x, file, line);
}

//**************************************************
//	extern var
//**************************************************
CDLogger OLogger;
