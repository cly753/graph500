
#ifndef VERTEX_RELABEL_H
#define VERTEX_RELABEL_H

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stddef.h>

#include "common.h"

#include "constants.h"
#include "print.h"
#include "bfs.h"

extern int64_t max_index;
extern int64_t non_zero_degree_count;

int64_t filter_zero_degree(const tuple_graph* const tg);
void broadcast_filter_zero_degree_result();
int64_t get_new_index(int64_t old_index);
int64_t get_old_index(int64_t new_index);
void calculate_remapped_count();

int been_relabelled(int64_t old_index);

void recover_index(int64_t *pred);


#endif // VERTEX_RELABEL_H