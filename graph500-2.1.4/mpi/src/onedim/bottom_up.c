#include "bottom_up.h"

#include <string.h>

#include "oned_csr.h"

#include "constants.h"
#include "bfs.h"
#include "parent_tracker.h"
#include "print.h"

extern oned_csr_graph g;
extern int64_t *pred;

extern int64_t *parent_cur; // [parent sending to rank-0 (parent1, count1, parent2, count2) : parent sending to rank-1 : ... ]
extern int64_t *parent_next; // [parent sending to rank-0 (parent1, count1, parent2, count2) : parent sending to rank-1 : ... ]

extern int64_t *frontier;
extern int64_t *frontier_next;

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

void init_bottom_up() {
    if (frontier == NULL)
        frontier = xmalloc(global_long_nb);
    if (frontier_next == NULL)
        frontier_next = xmalloc(global_long_nb);

    memset(frontier_next, 0, global_long_nb);
}

int have_more_bottom_up() {
    int i;
    for (i = 0; i < global_long_n; i++)
        if (frontier[i])
            return 1;
    return 0;
}

void sync_bottom_up() {
#ifdef SHOWDEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    PRINTLN("rank %02d: frontier_next:", rank)
    show_global(frontier_next);
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    // allreduce to broadcast frontier
    MPI_Allreduce(
            frontier_next, // void* send_data
            frontier, // void* recv_data
            global_long_n, // int count
            MPI_LONG, // MPI_Datatype datatype
            MPI_BOR, // MPI_Op op
            MPI_COMM_WORLD // MPI_Comm communicator
    );
    memset(frontier_next, 0, global_long_nb);
#ifdef SHOWDEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    PRINTLN("rank %02d: frontier:", rank)
    show_global(frontier);
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

inline int in_frontier(int64_t node) {
    return TEST_GLOBAL(node, frontier);
}

void one_step_bottom_up() {
    int i;
    for (i = 0; i < g.nlocalverts; i++) {
#ifdef SHOWDEBUG
        PRINTLN_RANK("checking vertex %d ...", VERTEX_TO_GLOBAL(rank, i))
#endif
        if (pred[i] == -1) {
#ifdef SHOWDEBUG
            PRINTLN_RANK("checking vertex %d yes", VERTEX_TO_GLOBAL(rank, i))
#endif
            int j;
            for (j = (int) g.rowstarts[i]; j < g.rowstarts[i + 1]; j++) {
                int64_t parent_global = g.column[j];
#ifdef SHOWDEBUG
                PRINTLN_RANK("checking vertex %d - %d ...", VERTEX_TO_GLOBAL(rank, i), parent_global)
#endif
                if (in_frontier(parent_global)) {
#ifdef SHOWDEBUG
                    PRINTLN_RANK("checking vertex %d - %d yes", VERTEX_TO_GLOBAL(rank, i), parent_global)
#endif
                    pred[i] = parent_global;
                    SET_GLOBAL(VERTEX_TO_GLOBAL(rank, i), frontier_next);
                    break;
                }
#ifdef SHOWDEBUG
                else
                PRINTLN_RANK("checking vertex %d - %d no.", VERTEX_TO_GLOBAL(rank, i), parent_global)
#endif
            }
        }
    }
}

void switch_to_bottom_up() { // use after one-step
    sync();
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
            SET_GLOBAL(to, frontier_next);
        }
    }
}