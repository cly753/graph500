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

int root_owner;

double time_comm;
double time_comp;
double time_temp;

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
int now_top_down;
int nth_call;
const int cutoff_to_bottom_up = 2; // first topdown->bottomup switch
const int cutoff_to_top_down = 4; // first bottomup->topdown switch
const float alpha = 0.2;
const float beta = 10;
int top_down_better() {
    // so good!
    if (nth_call > cutoff_to_top_down)
        return 0;
    if (nth_call++ < cutoff_to_bottom_up)
        return 1;
    return 0;

    // cann't see benefit, need more test
    int nf = 0;
    int i;
    for (i = 0; i < global_long_n; i++)
        nf += __builtin_popcountll(frontier[i]);
    return nf < g.nglobalverts / beta;

    // too slow
    if (now_top_down) {
        int64_t mf = 0; // number of edges connecting frontier nodes
        int i;
        for (i = 0; i < g.nglobalverts; i++) {
            if (TEST_GLOBAL(i, frontier)) {
                mf += in_edge_start[i + 1] - in_edge_start[i];
            }
        }

        int64_t mu = 0; // number of edges connecting unvisited nodes
        for (i = 0; i < g.nlocalverts; i++) {
            if (pred[i] == -1) {
                mu += g.rowstarts[i + 1] - g.rowstarts[i];
            }
        }

        return mf < mu / alpha;
    }
    else {
        int nf = 0;
        int i;
        for (i = 0; i < global_long_n; i++)
            nf += __builtin_popcountll(frontier[i]);
        return nf < g.nglobalverts / beta;
    }
}

void init() {
    memset(pred, -1, g.nlocalverts * sizeof(int64_t));

    now_top_down = 1;
    nth_call = 0;
}

void wrap_up() {
#ifdef SHOWTIMER
    if (rank == root_owner) {
        double time_all = time_comm + time_comp;
        PRINTLN_RANK("time commp: %.6lfs (%.3f), time comm: %.6lfs (%.3f)",
                     time_comp, time_comp / time_all, time_comm, time_comm / time_all)
    }
#endif

    int64_t visit_node = 0;
    int i;
    for (i = 0; i < g.nlocalverts; i++)
        if (pred[i] != -1)
            visit_node++;

//    int64_t total_visit[1];
//    MPI_Reduce(
//        visit_node, // void* send_data,
//        &total_visit, // void* recv_data,
//        1, // int count,
//        MPI_LONG, // MPI_Datatype datatype,
//        MPI_SUM, // MPI_Op op,
//        0, // int root,
//        MPI_COMM_WORLD); // MPI_Comm communicator)
//    if (rank == 0)
//        PRINTLN_RANK("total visit: %d", &total_visit);
}

void bfs_gpu(int64_t root) {
    if (rank == root_owner) {
#ifdef SHOWDEBUG
        PRINTLN("rank %02d: root: %d", rank, (int)root)
#endif
        pred[VERTEX_LOCAL(root)] = root;
    }

    SET_GLOBAL(root, frontier);

    pred_to_gpu();

    while (1) {
        one_step_bottom_up_gpu();
#ifdef SHOWDEBUG
        show_pred();
#endif

        sync_frontier();

        if (!frontier_have_more())
            break;
    }

    pred_from_gpu();
}

void bfs(oned_csr_graph *gg, int64_t root, int64_t *predpred) {
    pred = predpred;
    init();
    init_frontier();

    root_owner = VERTEX_OWNER(root);

    bfs_gpu(root);
    return ;

#ifdef SHOWTIMER
    double level_start = 0;
    double level_stop = 0;
#endif

    if (rank == root_owner) {
#ifdef SHOWDEBUG
        PRINTLN_RANK("root: %d", rank, (int)root)
#endif
        pred[VERTEX_LOCAL(root)] = root;
    }

    SET_GLOBAL(root, frontier);

    while (1) {
#ifdef SHOWTIMER
        if (rank == root_owner)
            level_start = MPI_Wtime();
#endif

        if (top_down_better()) {
            TIME_IT(one_step_top_down(), &time_temp)
        }
        else {
            TIME_IT(one_step_bottom_up(), &time_temp)
        }
        time_comp += time_temp;

#ifdef SHOWDEBUG
        show_pred();
#endif

        TIME_IT(sync_frontier(), &time_temp)
        time_comm += time_temp;

        if (!frontier_have_more())
            break;

#ifdef SHOWTIMER
        if (rank == root_owner) {
            level_stop = MPI_Wtime();
            PRINTLN("[TIMER] %.6lfs", level_stop - level_start);
        }
#endif
    }

    wrap_up();
}

