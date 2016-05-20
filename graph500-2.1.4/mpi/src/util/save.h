
#ifndef SAVE_H
#define SAVE_H

#include "common.h"

#include <stdbool.h>

bool load(tuple_graph* tg, int scale, int degree, int64_t *bfs_roots, int num_bfs_roots);
bool save(tuple_graph* tg, int scale, int degree, int64_t *bfs_roots, int num_bfs_roots);

#endif // SAVE_H