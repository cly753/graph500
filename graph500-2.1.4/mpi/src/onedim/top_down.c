#include "top_down.h"

#include "bfs.h"
#include "print.h"

extern oned_csr_graph g;
extern int64_t *pred;

extern int64_t *frontier;
extern int64_t *frontier_next;

extern int64_t *in_edge_start;
extern int64_t *in_edge_to;

#ifdef FILTER_ZERO_DEGREE
extern int64_t non_zero_degree_count_global;
// extern int64_t non_zero_degree_count;
#endif

// int64_t *candidate;

void one_step_top_down() {
#ifdef USE_OPENMP
    omp_set_num_threads(12);
#endif

#ifdef SHOWDEBUG
    PRINTLN("--- top down ---")
#endif


    // if (candidate == NULL)
    //     candidate = xmalloc(local_long_nb);
    // memset(candidate, 0, local_long_nb);


    int i;

#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
#ifndef FILTER_ZERO_DEGREE
    for (i = 0; i < g.nglobalverts; i++) {
#else
    for (i = 0; i < non_zero_degree_count_global; i++) {
#endif
        if (!TEST_GLOBAL(i, frontier))
            continue;
        int j;
        for (j = (int)in_edge_start[i]; j < in_edge_start[i + 1]; j++) {
            int64_t to = in_edge_to[j];
            // SET_LOCAL_WITH_LOCAL(VERTEX_LOCAL(to), candidate);

            // if (!TEST_LOCAL_WITH_LOCAL(VERTEX_LOCAL(to), pred_visited)) {
            if (pred[VERTEX_LOCAL(to)] == -1) {
                pred[VERTEX_LOCAL(to)] = i;
                // SET_LOCAL_WITH_LOCAL(VERTEX_LOCAL(to), pred_visited);
#ifdef USE_OPENMP
                #pragma omp critical
#endif
                {
                    SET_GLOBAL(to, frontier_next);
                }
            }
        }
    }

// #ifndef FILTER_ZERO_DEGREE
//     for (i = 0; i < g.nlocalverts; i++) {
// #else
//     for (i = 0; i < non_zero_degree_count; i++) {
// #endif
//         if (TEST_LOCAL_WITH_LOCAL(i, candidate)) {
//             if (pred[i] == -1) {
//                 // SET_LOCAL_WITH_LOCAL(i, pred_visited);

//                 int j;
//                 for (j = g.rowstarts[i]; j < g.rowstarts[i + 1]; j++) {
//                     int64_t from = g.column[j];
//                     if (TEST_GLOBAL(from, frontier)) {
//                         pred[i] = from;
//                         // PRINTLN_RANK("from %d to %d, adding %d to frontier_next", from, VERTEX_TO_GLOBAL(rank, i), VERTEX_TO_GLOBAL(rank, i))
//                         SET_GLOBAL(VERTEX_TO_GLOBAL(rank, i), frontier_next);
//                     }
//                 }    
//             }
//         }
//     }
}