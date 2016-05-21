
#ifndef BFS_H
#define BFS_H

#include "limits.h"
#include "common.h"
#include "oned_csr.h"

#define LONG_BITS (sizeof(unsigned long) * CHAR_BIT)

#define SET_LOCAL(v, a) do {(a)[VERTEX_LOCAL((v)) / LONG_BITS] |= (1UL << (VERTEX_LOCAL((v)) % LONG_BITS));} while (0)
#define TEST_LOCAL(v, a) (((a)[VERTEX_LOCAL((v)) / LONG_BITS] & (1UL << (VERTEX_LOCAL((v)) % LONG_BITS))) != 0)

#define SET_GLOBAL(v, a) do {(a)[(v) / LONG_BITS] |= (1UL << ((v) % LONG_BITS));} while (0)
#define TEST_GLOBAL(v, a) (((a)[(v) / LONG_BITS] & (1UL << ((v) % LONG_BITS))) != 0)

int local_long_n;
size_t local_long_nb;
int global_long_n;
size_t global_long_nb;

//unsigned long *g_cur;
//unsigned long *g_next;


void bfs(oned_csr_graph* g, int64_t root, int64_t* pred);

#endif // BFS_H