
#ifndef SORT_GRAPH_H
#define SORT_GRAPH_H

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stddef.h>

#include "common.h"
#include "oned_csr.h"

#include "constants.h"
#include "print.h"
#include "bfs.h"

void sort_csr_by_degree(oned_csr_graph *g);

void sort_in_edge_by_degree(int64_t *in_edge_start, int64_t *in_edge_to);

#endif // SORT_GRAPH_H