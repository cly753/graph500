#include "common.h"
#include "oned_csr.h"
#include "oned_csc.h"
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
#include "build_graph.h"
// #include "relabel.h"
#include "vertex_relabel.h"
#include "sort_graph.h"

#include "../../../generator/graph_generator.h"



oned_csr_graph g;

void show_g() {
    REACH_HERE_RANK
    PRINTLN("rank %02d: nlocalverts: %"PRId64", max_nlocalverts: %"PRId64
        ", nlocaledges: %"PRId64", lg_nglobalverts: %d, nglobalverts: %"PRId64"",
        rank,
        g.nlocalverts, g.max_nlocalverts, g.nlocaledges, g.lg_nglobalverts, g.nglobalverts)
}

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

int64_t *in_edge_start;
int64_t *in_edge_to; // global index

void show_in_edge() {
    int i;
    for (i = 0; i < g.nglobalverts; i++) {
        PRINT_RANK("in edge: %d >", i)
        int j;
        for (j = in_edge_start[i]; j < in_edge_start[i + 1]; j++) {
            PRINT(" %d", (int)in_edge_to[j])
        }
        PRINTLN("")
    }
}

void csr_to_in_edge() {
    // size of in_edge_count_global array in bytes
    int64_t in_edge_count_global_b_size = g.nglobalverts * sizeof(int64_t);
    int64_t *in_edge_count_global = xmalloc(in_edge_count_global_b_size);
    memset(in_edge_count_global, 0, in_edge_count_global_b_size);

    int i;
    for (i = 0; i < g.nlocalverts; i++) {
        int j;
        for (j = g.rowstarts[i]; j < g.rowstarts[i + 1]; j++) {
            // [index]: index is global vertex index
            // so count how many edges pointing to local vertex
            in_edge_count_global[g.column[j]]++;
        }
    }

    // one more for end index of last vertex
    in_edge_start = xmalloc(in_edge_count_global_b_size + sizeof(int64_t));
    in_edge_start[0] = 0;
    for (i = 1; i <= g.nglobalverts; i++) {
        // accumulate count
        in_edge_start[i] = in_edge_start[i - 1] + in_edge_count_global[i - 1];
    }
    // for each local edge from-to
    // the index in in_edge_to at which the to vertex should be placed
    // reuse memory of in_edge_count_global
    int64_t *cur_fill_index = in_edge_count_global;
    memcpy(cur_fill_index, in_edge_start, in_edge_count_global_b_size);

    in_edge_to = xmalloc(g.rowstarts[g.nlocalverts] * sizeof(int64_t));
    for (i = 0; i < g.nlocalverts; i++) {
        int i_global = VERTEX_TO_GLOBAL(rank, i);
        int j;
        for (j = g.rowstarts[i]; j < g.rowstarts[i + 1]; j++) {
            // the other end of the edge
            // global index
            int64_t to = g.column[j];
            in_edge_to[cur_fill_index[to]] = i_global;
            cur_fill_index[to]++;
        }
    }
    // cur_fill_index's memory is same as in_edge_count_global
    free(in_edge_count_global);
}

void show_csr() {
    int i;
    for (i = 0; i < g.nlocalverts; i++) {
        int j;
        PRINT_RANK("vertex %d >", (int)(VERTEX_TO_GLOBAL(rank, i)))
        for (j = g.rowstarts[i]; j < g.rowstarts[i + 1]; j++) {
            PRINT(" [%d]%d", j, (int)g.column[j])
        }
        PRINTLN("")
    }
}

void filter_duplicate_edge() {
    int64_t *dest_exist = xmalloc(global_long_nb);
    int new_j = 0;
    int new_start = -1;
    int i;
    for (i = 0; i < g.nlocalverts; i++) {
        memset(dest_exist, 0, global_long_nb);
        new_start = new_j;
        int j;
        for (j = g.rowstarts[i]; j < g.rowstarts[i + 1]; j++) {
            if (!TEST_GLOBAL(g.column[j], dest_exist)) {
                SET_GLOBAL(g.column[j], dest_exist);
                g.column[new_j] = g.column[j];
                new_j++;
            }
        }
        g.rowstarts[i] = new_start;
    }
    g.rowstarts[i] = new_j;
    free(dest_exist);
}

int64_t non_zero_degree_count;

void count_duplicate_edge() {
    int64_t *dest_exist = xmalloc(global_long_nb);
    int i;
    int total_duplicated_edge = 0;
    int total_edge = g.rowstarts[g.nlocalverts] - g.rowstarts[0];
    int zero_edge_vertex = 0;
    for (i = 0; i < g.nlocalverts; i++) {
        memset(dest_exist, 0, global_long_nb);
        int duplicated_edge = 0;
        int j;
        for (j = g.rowstarts[i]; j < g.rowstarts[i + 1]; j++) {
            if (TEST_GLOBAL(g.column[j], dest_exist))
                duplicated_edge++;
            else
                SET_GLOBAL(g.column[j], dest_exist);
        }
        total_duplicated_edge += duplicated_edge;
        // int64_t global_i = VERTEX_TO_GLOBAL(rank, i);
        int edge = g.rowstarts[i + 1] - g.rowstarts[i];
        if (edge == 0)
            zero_edge_vertex++;
        // PRINTLN_RANK("vertex %d have %d duplicated edges (total %d edges) (%3f).",
        //        (int)global_i, duplicated_edge, edge, duplicated_edge / (float)edge)
    }
    PRINTLN_RANK("g.nlocaledges: %"PRId64"", g.nlocaledges)
    PRINTLN_RANK("summary: %d vertex have %d duplicated edges (total %d edges) (%3f), %d zero edge vertex (%3f).",
                 (int)g.nlocalverts, total_duplicated_edge, total_edge,
                 total_duplicated_edge / (float)total_edge, zero_edge_vertex,
                 zero_edge_vertex / (float)g.nlocalverts)
    free(dest_exist);

    non_zero_degree_count = g.nlocalverts - zero_edge_vertex;
}

void make_graph_data_structure(const tuple_graph* const tg) {
    PRINTLN_RANK("tg->edgememory_size=%"PRId64"", tg->edgememory_size)

#ifndef FILTER_ZERO_DEGREE
#ifdef NEW_GRAPH_BUILDER
    new_convert_graph_to_oned_csr(tg, &g);
#else
    convert_graph_to_oned_csr(tg, &g);
#endif
#ifdef SHOWDEBUG
    show_csr();
#endif
    
#else // FILTER_ZERO_DEGREE
    tuple_graph *tg_copy = xmalloc(sizeof(tuple_graph));
    *tg_copy = *tg;
    tg_copy->edgememory = xmalloc(tg->edgememory_size * sizeof(packed_edge));
    memcpy(tg_copy->edgememory, tg->edgememory, tg->edgememory_size * sizeof(packed_edge));

    filter_zero_degree(tg_copy);
    broadcast_filter_zero_degree_result();

#ifdef NEW_GRAPH_BUILDER
    new_convert_graph_to_oned_csr(tg_copy, &g);
#else
    convert_graph_to_oned_csr(tg_copy, &g);
#endif

    free(tg_copy->edgememory);
    free(tg_copy);
#endif // FILTER_ZERO_DEGREE

    local_long_n = (g.nlocalverts + LONG_BITS - 1) / LONG_BITS;
    local_long_nb = local_long_n * sizeof(unsigned long);
    global_long_n = (g.nglobalverts + LONG_BITS - 1) / LONG_BITS;
    global_long_nb = global_long_n * sizeof(unsigned long);

#ifdef FILTER_EDGE
#ifdef SHOWDEBUG
    show_csr();
    count_duplicate_edge();
#endif
    filter_duplicate_edge();

#ifdef SHOWDEBUG
    show_csr();
#endif
#endif

#if defined(SHOWDEBUG) || defined(FILTER_ZERO_DEGREE)
    count_duplicate_edge();
#endif

    csr_to_in_edge();
#ifdef SHOWDEBUG
   show_in_edge();
#endif

#ifdef FILTER_ZERO_DEGREE
    calculate_remapped_count();
#endif

    sort_csr_by_degree(&g);
    sort_in_edge_by_degree(in_edge_start, in_edge_to);

#ifdef SHOWDEBUG
    show_g();
#endif
}

void free_graph_data_structure(void) {
    free_oned_csr_graph(&g);

    // TODO free others
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

#ifndef FILTER_ZERO_DEGREE

    bfs(&g, root, pred);

#else // FILTER_ZERO_DEGREE
    if (!been_relabelled(root)) {
        memset(pred, -1, g.nlocalverts * sizeof(int64_t));
        if (rank == VERTEX_OWNER(root))
            pred[VERTEX_LOCAL(root)] = root;
    }
    else {
        int64_t new_root = get_new_index(root);
#ifdef SHOWDEBUG
        PRINTLN_RANK("get_new_index(%"PRId64") = %"PRId64"", root, new_root)
#endif
        bfs(&g, new_root, pred);

#ifdef SHOWDEBUG
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            PRINTLN_RANK("BEFORE RECOVERING")
            show_pred();
            MPI_Barrier(MPI_COMM_WORLD);
        }
        else {
            MPI_Barrier(MPI_COMM_WORLD);
            PRINTLN_RANK("BEFORE RECOVERING")
            show_pred();
        }
        MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef SHOWTIMER
        double t_start = 0;
        double t_stop = 0;
        double t_total = 0;
        if (rank == 0)
            t_start = MPI_Wtime();
#endif

#ifndef PUT_RECOVER_ZERO_DEGREE_IN_VALIDATION
        recover_index(pred);

#ifdef SHOWTIMER
        if (rank == 0) {
            t_stop = MPI_Wtime();
            t_total = t_stop - t_start;
            PRINTLN("[TIMER] time for recover_index: %.6lfs", t_total);
        }
#endif
#endif
    }    

#ifdef SHOWDEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        PRINTLN_RANK("AFTER RECOVERING")
        show_pred();
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else {
        MPI_Barrier(MPI_COMM_WORLD);
        PRINTLN_RANK("AFTER RECOVERING")
        show_pred();
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif // FILTER_ZERO_DEGREE
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
