
#include "print.h"

void print_binary_int(unsigned int value, char* result) {
    int len = 32;
    result[len] = 0;
    int i;
    for (i = 0; i < len; i++) {
        result[i] = (value & (1 << i)) ? '1' : '0';
    }
}

void print_binary_long(unsigned long value, char* result) {
    int len = 64;
    result[len] = 0;
    int i;
    for (i = 0; i < len; i++) {
        result[i] = (value & (1UL << i)) ? '1' : '0';
    }
}
