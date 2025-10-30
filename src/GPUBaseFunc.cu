#include"GPUBaseFunc.h"

// strrchr 函数
__device__ const char* strrchr_GPUd(const char* str, int c) {
    const char* found = nullptr;
    while (*str != '\0') {
        if (*str == c) {
            found = str; // 记录最后一次出现的位置
        }
        str++;
    }
    // 检查字符串结束符 '\0' 是否也是指定要找的字符
    if (c == '\0') {
        return str;
    }
    return found; // 如果没找到，返回 nullptr
}
__device__ char* strrchr_GPUd(char* str, int c) {
    // 通过const_cast调用const版本，避免代码重复
    const char* result = strrchr_GPUd(const_cast<const char*>(str), c);
    return const_cast<char*>(result);
}

__device__ char* uint64_to_string(uint64_t value, char* str, int base) {
    char* ptr = str;
    char* low = str;

    if (base < 2 || base > 36) {
        *ptr = '\0';
        return str;
    }

    do {
        int digit = value % base;
        *ptr++ = (digit < 10) ? '0' + digit : 'a' + digit - 10;
        value /= base;
    } while (value);

    *ptr = '\0';

    // 反转字符串
    ptr--;
    while (low < ptr) {
        char tmp = *low;
        *low++ = *ptr;
        *ptr-- = tmp;
    }

    return str;
}

__device__ char* to_string(size_t value, char* str) {
    return uint64_to_string(value, str, 10);
}

// 字符串拼接函数
__device__ char* strMerge(const char* str1, const char* str2, char* output) {
    char* ptr = output;

    // 复制第一个字符串
    while (*str1 != '\0') {
        *ptr++ = *str1++;
    }

    // 复制第二个字符串
    while (*str2 != '\0') {
        *ptr++ = *str2++;
    }

    // 添加字符串结束符
    *ptr = '\0';

    return output;
}

// 格式化索引信息函数
__device__ char* indexInfoToString(size_t a1, size_t a2, char* output) {
    // 转换数字为字符串
    char a1_str[32];
    char a2_str[32];
    to_string(a1, a1_str);
    to_string(a2, a2_str);

    // 构建前缀和后缀
    const char prefix[] = "index=";
    const char middle[] = ", capacity=";

    // 分步拼接字符串
    char temp[64];
    strMerge(prefix, a1_str, temp);
    strMerge(temp, middle, temp);
    strMerge(temp, a2_str, output);

    return output;
}