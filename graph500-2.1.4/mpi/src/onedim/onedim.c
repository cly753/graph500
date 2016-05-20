#include "common.h"
#include "oned_csr.h"
#include <mpi.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
#include <assert.h>

#include "constants.h"
#include "print.h"
#include "bfs.h"

oned_csr_graph g;
// typedef struct oned_csr_graph {
//     size_t nlocalverts;
//     int64_t max_nlocalverts;
//     size_t nlocaledges;
//     int lg_nglobalverts;
//     int64_t nglobalverts;
//     size_t *rowstarts;
//     int64_t *column;
//     const tuple_graph* tg; /* Original graph used to build this one */
// } oned_csr_graph;
// 
// typedef struct packed_edge {
//   uint32_t v0_low;
//   uint32_t v1_low;
//   uint32_t high; /* v1 in high half, v0 in low half */
// } packed_edge;
// 
// typedef struct tuple_graph {
//     int data_in_file;
//     /* 1 for file, 0 for memory */
//     packed_edge *restrict edgememory;
//     /* NULL if edges are in file */
//     int64_t edgememory_size;
//     int64_t max_edgememory_size;
//     MPI_File edgefile;
//     /* Or MPI_FILE_NULL if edges are in memory */
//     int64_t nglobaledges; /* Number of edges in graph, in both cases */
// } tuple_graph;


void make_graph_data_structure(const tuple_graph* const tg) {
    convert_graph_to_oned_csr(tg, &g);

    local_long_n = (g.nlocalverts + LONG_BITS - 1) / LONG_BITS;
    local_long_nb = local_long_n * sizeof(unsigned long);
    global_long_n = (g.nglobalverts + LONG_BITS - 1) / LONG_BITS;
    global_long_nb = global_long_n * sizeof(unsigned long);

//    g_cur = (unsigned long) xmalloc(global_long_nb);
//    g_next = (unsigned long) xmalloc(global_long_nb);
}

void free_graph_data_structure(void) {
    free_oned_csr_graph(&g);

//    free(g_cur);
//    free(g_next);
}

int bfs_writes_depth_map(void) {
    /* Change to 1 if high 16 bits of each entry of pred are the (zero-based) BFS
    * level number, with UINT16_MAX for unreachable vertices. */
    return 0;
}


void run_bfs(int64_t root, int64_t* pred) {
    /* Predefined entities you can use in your BFS (from common.h and oned_csr.h):
    *   + rank: global variable containing MPI rank
    *   + size: global variable containing MPI size
    *   + DIV_SIZE: single-parameter macro that divides by size (using a shift
    *     when properly set up)
    *   + MOD_SIZE: single-parameter macro that reduces modulo size (using a
    *     mask when properly set up)
    *   + VERTEX_OWNER: single-parameter macro returning the owner of a global
    *     vertex number
    *   + VERTEX_LOCAL: single-parameter macro returning the local offset of a
    *     global vertex number
    *   + VERTEX_TO_GLOBAL: single-parameter macro converting a local vertex
    *     offset to a global number
    *   + g.nlocalverts: number of vertices stored on the local rank
    *   + g.nglobalverts: total number of vertices in the graph
    *   + g.nlocaledges: number of graph edges stored locally
    *   + g.rowstarts, g.column: zero-based compressed sparse row data
    *     structure for the local part of the graph
    *
    * All macros documented above evaluate their arguments exactly once.
    *
    * The graph is stored using a 1-D, cyclic distribution: all edges incident
    * to vertex v are stored on rank (v % size) (aka VERTEX_OWNER(v)).  Edges
    * that are not self-loops are stored twice, once for each endpoint;
    * duplicates edges are kept.  The neighbors of vertex v can be obtained on
    * rank VERTEX_OWNER(v); they are stored in elements
    * {g.rowstarts[VERTEX_LOCAL(v)] ... g.rowstarts[VERTEX_LOCAL(v) + 1] - 1}
    * (inclusive) of g.column.
    *
    * Upon exit, your BFS must have filled in:
    *   + pred (an array of size g.nlocalverts):
    *     - The predecessor of vertex v in the BFS tree should go into
    *       pred[VERTEX_LOCAL(v)] on rank VERTEX_OWNER(v)
    *     - The predecessor of root is root
    *     - The predecessor of any unreachable vertex is -1
    *
    * The validator will check this for correctness. */

#ifdef SHOWDEBUG
    PRINTLN("rank %02d: nlocalverts: %"PRId64", max_nlocalverts: %"PRId64
        ", nlocaledges: %"PRId64", lg_nglobalverts: %d, nglobalverts: %"PRId64"",
        rank,
        g.nlocalverts, g.max_nlocalverts, g.nlocaledges, g.lg_nglobalverts, g.nglobalverts)
#endif
    bfs(&g, root, pred);
}

void get_vertex_distribution_for_pred(size_t count, const int64_t* vertex_p, int* owner_p, size_t* local_p) {
    const int64_t* restrict vertex = vertex_p;
    int* restrict owner = owner_p;
    size_t* restrict local = local_p;
    ptrdiff_t i;
#pragma omp parallel for
    for (i = 0; i < (ptrdiff_t)count; ++i) {
        owner[i] = VERTEX_OWNER(vertex[i]);
        local[i] = VERTEX_LOCAL(vertex[i]);
    }
}

int64_t vertex_to_global_for_pred(int v_rank, size_t v_local) {
    return VERTEX_TO_GLOBAL(v_rank, v_local);
}

size_t get_nlocalverts_for_pred(void) {
    return g.nlocalverts;
}
