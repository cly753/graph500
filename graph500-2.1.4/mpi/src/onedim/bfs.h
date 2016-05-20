
#ifndef BFS_H
#define BFS_H

#include "limits.h"
#include "common.h"
#include "oned_csr.h"

#define LONG_BITS (sizeof(unsigned long) * CHAR_BIT)

int64_t local_long_n;
int64_t local_long_nb;
int64_t global_long_n;
int64_t global_long_nb;

//unsigned long *g_cur;
//unsigned long *g_next;


void bfs(oned_csr_graph* g, int64_t root, int64_t* pred);

#endif // BFS_H