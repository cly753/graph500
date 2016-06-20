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
#endif

void one_step_top_down() {
#ifdef USE_OPENMP
    omp_set_num_threads(2);
#endif
    
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
            if (pred[VERTEX_LOCAL(to)] == -1) {
                pred[VERTEX_LOCAL(to)] = i;
#ifdef USE_OPENMP
                #pragma omp critical
#endif
                {
                    SET_GLOBAL(to, frontier_next);
                }
            }
        }
    }
}