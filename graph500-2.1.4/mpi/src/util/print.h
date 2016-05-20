#ifndef PRINT_H
#define PRINT_H

#ifdef REACH_HERE
#undef REACH_HERE
#endif

#ifdef PRINT
#undef PRINT
#endif

#ifdef PRINTLN
#undef PRINTLN
#endif

#ifdef SHOWINT
#undef SHOWINT
#endif

#ifdef SHOWLONG
#undef SHOWLONG
#endif

#ifdef SHOWINTB
#undef SHOWINTB
#endif

#ifdef SHOWLONGB
#undef SHOWLONGB
#endif

#include <assert.h>

//#define SHOWDEBUG

#define DOPRINT
#ifdef DOPRINT

#define REACH_HERE { fprintf(stderr, "REACH_HERE %s:%d\n", __FUNCTION__, __LINE__); fflush(stderr); }
#define PRINT(s, ...) { fprintf(stderr, s, ##__VA_ARGS__); fflush(stderr); }
#define PRINTLN(s, ...) { fprintf(stderr, s "\n", ##__VA_ARGS__); fflush(stderr); }

#define SHOWINT(x) { fprintf(stderr, "%s = %d\n", #x, (x)); fflush(stderr); }
#define SHOWLONG(x) { fprintf(stderr, "%s = %"PRId64"\n", #x, (x)); fflush(stderr); }
#define SHOWINTB(x) { \
    assert(sizeof(x) == sizeof(unsigned int)); \
    char s[33]; \
    print_binary_int(x, &s[0]); \
    fprintf(stderr, "%s = %s\n", #x, s); \
    fflush(stderr); }

#define SHOWLONGB(x) { \
    assert(sizeof(x) == sizeof(unsigned long)); \
    char s[65]; \
    print_binary_long(x, &s[0]); \
    fprintf(stderr, "%s = %s\n", #x, s); \
    fflush(stderr); }

#else

#define REACH_HERE {;}
#define PRINT(s, ...) {;}
#define PRINTLN(s, ...) {;}

#define SHOWINT(x) {;}
#define SHOWLONG(x) {;}
#define SHOWINTB(x) {;}

#define SHOWLONGB(x) {;}

#endif

void print_binary_int(unsigned int value, char* result);
void print_binary_long(unsigned long value, char* result);

#endif // PRINT_H


// how to use:
// 
// #define REACH_HERE {;}
// #define PRINT(s, ...) {;}
// #define PRINTLN(s, ...) {;}
// 
// // #undef HHHDEBUG
// #ifdef HHHDEBUG
// #include "print.h"
// #endif