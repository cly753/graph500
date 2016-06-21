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

int root_owner;

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

extern int64_t *in_edge_start;
extern int64_t *in_edge_to;
int nth_call;

// n == 20
// int top_down_better() {
//     switch (nth_call++) {
//     case 0 ... 1: // inclusive
//         return 1;
//     case 2 ... 6:
//         return 0;
//     default: // > 4
//         return 1;
//     }
// }

// n == 18
int top_down_better() {
    switch (nth_call++) {
    case 0 ... 1: // inclusive
        return 1;
    case 2 ... 4:
        return 0;
    default: // > 4
        return 1;
    }
}

void init() {
    memset(pred, -1, g.nlocalverts * sizeof(int64_t));

    nth_call = 0;

    // if (pred_visited == NULL)
    //     pred_visited = xmalloc(local_long_nb);
    // memset(pred_visited, 0, local_long_nb);
}

void bfs(oned_csr_graph *gg, int64_t root, int64_t *predpred) {
    pred = predpred;
    init();
    init_frontier();

    root_owner = VERTEX_OWNER(root);

#ifdef SHOWTIMER
    double t_start = 0;
    double t_stop = 0;
    double t_total = 0;
    int level = 0;
#endif

    if (rank == root_owner) {
#ifdef SHOWDEBUG
        PRINTLN_RANK("root: %d", (int)root)
#endif
        pred[VERTEX_LOCAL(root)] = root;
        // SET_LOCAL_WITH_LOCAL(VERTEX_LOCAL(root), pred_visited);
    }

    SET_GLOBAL(root, frontier);

    while (1) {
#ifdef SHOWTIMER
        if (rank == root_owner)
            t_start = MPI_Wtime();
#endif

        if (top_down_better())
            one_step_top_down();
        else
            one_step_bottom_up();

#ifdef SHOWDEBUG
        show_pred();
#endif

#ifdef SHOWTIMER
        double s_start;
        if (rank == root_owner) s_start = MPI_Wtime();
#endif

        sync_frontier();

#ifdef SHOWTIMER
        double s_stop;
        if (rank == root_owner) {
            s_stop = MPI_Wtime();
            PRINTLN("[TIMER] time for comm : %.6lfs", s_stop - s_start);

            t_stop = MPI_Wtime();
            t_total = t_stop - t_start;

            PRINTLN("[TIMER] time for lvl %d: %.6lfs", level++, t_total);
        }
#endif

        if (!frontier_have_more())
            break;
    }
}

