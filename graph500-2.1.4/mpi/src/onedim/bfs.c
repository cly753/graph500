#include "bfs.h"

#include <stdlib.h>
#include <string.h>

#include "constants.h"
#include "print.h"
#include "top_down.h"
#include "bottom_up.h"
#include "frontier_tracker.h"

extern oned_csr_graph g;
int64_t *pred;

extern int64_t *parent_cur;

void show_local(int64_t *a) {
    int i;
    char s[65];
    for (i = 0; i < local_long_n; i++) {
        print_binary_long(a[i], s);
        PRINT("%s ", s);
    }
    PRINTLN("")
}

void show_global(int64_t *a) {
    int i;
    char s[65];
    for (i = 0; i < global_long_n; i++) {
        print_binary_long(a[i], s);
        PRINT("%s ", s);
    }
    PRINTLN("")
}

void show_pred() {
    int i;
    PRINT_RANK("index:")
    for (i = 0; i < g.nlocalverts; i++) {
        PRINT(" %2d", (int) VERTEX_TO_GLOBAL(rank, i))
    }
    PRINTLN("")
    PRINT_RANK("pred :")
    for (i = 0; i < g.nlocalverts; i++) {
        PRINT(" %2d", (int) pred[i])
    }
    PRINTLN("")
}

int bottom_up_better() {
//    int64_t mf = 0; // number of edges connecting frontier nodes
//    int64_t mu = 0; // number of edges connecting unvisited nodes

    // TODO
    // see Direction-Optimizing Breadth-First Search
    // for how to make decision

    return 0;
}

void init() {
    memset(pred, -1, g.nlocalverts * sizeof(int64_t));
}

void bfs(oned_csr_graph *gg, int64_t root, int64_t *predpred) {
    pred = predpred;
    init();
    init_frontier();

    int root_owner = VERTEX_OWNER(root);

#ifdef SHOWTIMER
    double level_start = 0;
    double level_stop = 0;
    if (rank == root_owner)
        level_start = MPI_Wtime();
#endif

    if (rank == root_owner) {
#ifdef SHOWDEBUG
        PRINTLN("rank %02d: root: %d", rank, (int)root)
#endif

        pred[VERTEX_LOCAL(root)] = root;
    }

    SET_GLOBAL(root, frontier);
#ifdef BOTTOM_UP
    one_step_bottom_up();
#else
    one_step_top_down();
#endif

#ifdef SHOWDEBUG
    show_pred();
#endif

#ifdef SHOWTIMER
    if (rank == root_owner) {
        level_stop = MPI_Wtime();
        PRINTLN("[TIMER] %.6lfs", level_stop - level_start);
    }
#endif

    while (1) {
#ifdef SHOWTIMER
        if (rank == root_owner)
            level_start = MPI_Wtime();
#endif

        sync_frontier();
        if (!frontier_have_more())
            break;
#ifndef BOTTOM_UP
        one_step_top_down();
#else
        one_step_bottom_up();
#endif

#ifdef SHOWDEBUG
        show_pred();
#endif
#ifdef SHOWTIMER
        if (rank == root_owner) {
            level_stop = MPI_Wtime();
            PRINTLN("[TIMER] %.6lfs", level_stop - level_start);
        }
#endif
    }
}

