#include "bottom_up.h"

#include "oned_csr.h"

#include "constants.h"
#include "bfs.h"
#include "print.h"

#ifdef FILTER_ZERO_DEGREE
extern int64_t non_zero_degree_count;
#endif

extern oned_csr_graph g;
extern int64_t *pred;

extern int64_t *frontier;
extern int64_t *frontier_next;

void one_step_bottom_up() {
    int i;

#ifdef USE_OPENMP
    omp_set_num_threads(2);
#endif
    
#ifdef USE_OPENMP
   #pragma omp parallel for
#endif
#ifndef FILTER_ZERO_DEGREE
    for (i = 0; i < g.nlocalverts; i++) {
#else
    for (i = 0; i < non_zero_degree_count; i++) {
#endif
        if (pred[i] == -1) {
            int j;
            for (j = (int) g.rowstarts[i]; j < g.rowstarts[i + 1]; j++) {
                int64_t parent_global = g.column[j];
                if (TEST_GLOBAL(parent_global, frontier)) {
                    pred[i] = parent_global;
                    SET_GLOBAL(VERTEX_TO_GLOBAL(rank, i), frontier_next);
                    break;
                }
            }
        }
    }
}
