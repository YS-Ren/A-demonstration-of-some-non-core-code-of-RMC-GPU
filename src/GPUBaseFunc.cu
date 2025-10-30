#include"GPUBaseFunc.h"

// strrchr ����
__device__ const char* strrchr_GPUd(const char* str, int c) {
    const char* found = nullptr;
    while (*str != '\0') {
        if (*str == c) {
            found = str; // ��¼���һ�γ��ֵ�λ��
        }
        str++;
    }
    // ����ַ��������� '\0' �Ƿ�Ҳ��ָ��Ҫ�ҵ��ַ�
    if (c == '\0') {
        return str;
    }
    return found; // ���û�ҵ������� nullptr
}
__device__ char* strrchr_GPUd(char* str, int c) {
    // ͨ��const_cast����const�汾����������ظ�
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

    // ��ת�ַ���
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

// �ַ���ƴ�Ӻ���
__device__ char* strMerge(const char* str1, const char* str2, char* output) {
    char* ptr = output;

    // ���Ƶ�һ���ַ���
    while (*str1 != '\0') {
        *ptr++ = *str1++;
    }

    // ���Ƶڶ����ַ���
    while (*str2 != '\0') {
        *ptr++ = *str2++;
    }

    // ����ַ���������
    *ptr = '\0';

    return output;
}

// ��ʽ��������Ϣ����
__device__ char* indexInfoToString(size_t a1, size_t a2, char* output) {
    // ת������Ϊ�ַ���
    char a1_str[32];
    char a2_str[32];
    to_string(a1, a1_str);
    to_string(a2, a2_str);

    // ����ǰ׺�ͺ�׺
    const char prefix[] = "index=";
    const char middle[] = ", capacity=";

    // �ֲ�ƴ���ַ���
    char temp[64];
    strMerge(prefix, a1_str, temp);
    strMerge(temp, middle, temp);
    strMerge(temp, a2_str, output);

    return output;
}