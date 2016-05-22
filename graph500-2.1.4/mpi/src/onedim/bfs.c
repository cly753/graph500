#include "bfs.h"

#include <stdlib.h>

#include "constants.h"
#include "print.h"
#include "bottom_up.h"
#include "parent_tracker.h"

extern oned_csr_graph g;
int64_t *pred;

extern int64_t *parent_cur;

void show_pred() {
    int i;
    PRINT("rank %02d: index:", rank)
    for (i = 0; i < g.nlocalverts; i++) {
        PRINT(" %2d", VERTEX_TO_GLOBAL(rank, i))
    }
    PRINTLN("")
    PRINT("rank %02d: pred :", rank)
    for (i = 0; i < g.nlocalverts; i++) {
        PRINT(" %2d", pred[i])
    }
    PRINTLN("")
}

int bottom_up_better() {
    int64_t mf = 0; // number of edges connecting frontier nodes
    int64_t mu = 0; // number of edges connecting unvisited nodes
    return 0;
}

void one_step() {
    int i;
    for (i = 0; parent_cur[i] != -1; i++) {
        int64_t from = parent_cur[i];
        i++;
        int64_t to = parent_cur[i];
        int64_t to_local = VERTEX_LOCAL(to);
#ifdef SHOWDEBUG
        PRINTLN("rank %02d: from %d to %d (local %d)", rank, (int)from, (int)to, (int)to_local)
#endif
        if (pred[to_local] == -1) {
            pred[to_local] = from;
            int j;
            for (j = g.rowstarts[to_local]; j < g.rowstarts[to_local + 1]; j++) {
                int64_t next_to = g.column[j];
                add_parent(to, next_to);
            }
        }
    }
}

void init() {
    memset(pred, -1, g.nlocalverts * sizeof(int64_t));
}

void bfs(oned_csr_graph *gg, int64_t root, int64_t *predpred) {
    init_parent_tracker();
    pred = predpred;
    init();

#ifdef BOTTOM_UP
    init_bottom_up();
#endif


    int root_owner = VERTEX_OWNER(root);

#ifdef SHOWTIMER
    double level_start;
    double level_stop;
    if (rank == root_owner)
        level_start = MPI_Wtime();
#endif

    if (rank == root_owner) {
#ifdef SHOWDEBUG
        PRINTLN("rank %02d: root: %d", rank, (int)root)
#endif
#ifdef BOTTOM_UP
        pred[VERTEX_LOCAL(root)] = root;
        SET_GLOBAL(root, frontier_next);
#else
        parent_cur[0] = parent_cur[1] = root;
        one_step();
#endif
#ifdef SHOWDEBUG
        show_parent();
        show_counter();
        show_pred();
#endif
    }

#ifdef SHOWTIMER
    if (rank == root_owner) {
        level_stop = MPI_Wtime();
        PRINTLN("[TIMER] %.6lfs", level_stop - level_start);
    }
#endif

#ifdef BOTTOM_UP
    while (1) {
#ifdef SHOWTIMER
        if (rank == root_owner)
            level_start = MPI_Wtime();
#endif
        sync_bottom_up();
        if (!have_more_bottom_up())
            break;
        one_step_bottom_up();
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
#else
    while (1) {
#ifdef SHOWTIMER
        if (rank == root_owner)
            level_start = MPI_Wtime();
#endif
        if (!have_more())
            break;
        sync();
        one_step();
#ifdef SHOWTIMER
        if (rank == root_owner) {
            level_stop = MPI_Wtime();
            PRINTLN("[TIMER] %.6lfs", level_stop - level_start);
        }
#endif
#ifdef SHOWDEBUG
        PRINTLN("rank %02d: one more level", rank)
        show_parent();
        show_counter();
        show_pred();
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
#endif
}

