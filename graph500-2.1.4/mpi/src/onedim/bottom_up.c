#include "bottom_up.h"

#include <string.h>

#include "oned_csr.h"

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
}

inline int in_frontier(int64_t node) {
    return TEST_GLOBAL(node, frontier);
}

void one_step_bottom_up() {
    int i;
    for (i = 0; i < g.nlocalverts; i++) {
        if (pred[i] == -1) {
            int j;
            for (j = (int) g.rowstarts[i]; j < g.rowstarts[i + 1]; j++) {
                int64_t parent_global = g.column[j];
                if (in_frontier(parent_global)) {
                    pred[i] = parent_global;
                    SET_GLOBAL(VERTEX_TO_GLOBAL(rank, i), frontier_next);
                    break;
                }
            }
        }
    }
}