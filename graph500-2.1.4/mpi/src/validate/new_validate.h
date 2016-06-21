#include "onesided.h"
#include "common.h"
#include <mpi.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

int new_validate_bfs_result(const tuple_graph* const tg, const int64_t nglobalverts, const size_t nlocalverts, const int64_t root, int64_t* const pred, int64_t* const edge_visit_count_ptr);
