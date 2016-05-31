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

void wrap_up() {
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

#define USE_TOPDOWN 1
#define USE_BOTTOMUP 2
#define USE_GPU 3

extern int64_t *in_edge_start;
extern int64_t *in_edge_to;
int now_top_down;
int nth_call;
const int cutoff_to_bottom_up = 2; // first topdown->bottomup switch
const int cutoff_to_top_down = 4; // first bottomup->topdown switch
const float alpha = 0.2;
const float beta = 10;
int get_strategy() {
    // return USE_GPU;
    int strategy = -1;
    switch (nth_call) {
    case 0 ... 1: // inclusive
        strategy = USE_TOPDOWN;
        break;
    case 2 ... 2:
        strategy = USE_GPU;
        break;
    case 3 ... 4:
        strategy = USE_BOTTOMUP;
        break;
    default: // > 4
        strategy = USE_TOPDOWN;
    }
    nth_call++;
    return strategy;

    // // cann't see benefit, need more test
    // int nf = 0;
    // int i;
    // for (i = 0; i < global_long_n; i++)
    //     nf += __builtin_popcountll(frontier[i]);
    // return nf < g.nglobalverts / beta;

    // // too slow
    // if (now_top_down) {
    //     int64_t mf = 0; // number of edges connecting frontier nodes
    //     int i;
    //     for (i = 0; i < g.nglobalverts; i++) {
    //         if (TEST_GLOBAL(i, frontier)) {
    //             mf += in_edge_start[i + 1] - in_edge_start[i];
    //         }
    //     }

    //     int64_t mu = 0; // number of edges connecting unvisited nodes
    //     for (i = 0; i < g.nlocalverts; i++) {
    //         if (pred[i] == -1) {
    //             mu += g.rowstarts[i + 1] - g.rowstarts[i];
    //         }
    //     }

    //     return mf < mu / alpha;
    // }
    // else {
    //     int nf = 0;
    //     int i;
    //     for (i = 0; i < global_long_n; i++)
    //         nf += __builtin_popcountll(frontier[i]);
    //     return nf < g.nglobalverts / beta;
    // }
}

void init() {
    memset(pred, -1, g.nlocalverts * sizeof(int64_t));

    now_top_down = 1;
    nth_call = 0;
}

void one_step_gpu_from_cpu() {
    save_frontier_g();
    pred_to_gpu();
    one_step_bottom_up_gpu();
    pred_from_gpu();
    read_frontier_next_g();
}

int frontier_have_more_gpu_from_cpu() {
    // TODO
    assert(0); 
}

void bfs(oned_csr_graph *gg, int64_t root, int64_t *predpred) {
    pred = predpred;
    init(); // set pred to -1
    init_frontier(); // set frontier, frontier_next to 0

    root_owner = VERTEX_OWNER(root);

#ifdef SHOWTIMER
    double t_start = 0;
    double t_stop = 0;
    double t_total = 0;
#endif

    if (rank == root_owner) {
#ifdef SHOWDEBUG
        PRINTLN_RANK("root: %d", rank, (int)root)
#endif
        pred[VERTEX_LOCAL(root)] = root;
    }
    // pred_to_gpu(); // ...
    SET_GLOBAL(root, frontier);

    while (1) {
#ifdef SHOWTIMER
        if (rank == root_owner)
            t_start = MPI_Wtime();
#endif

        int strategy = get_strategy();
        switch (strategy) {
        case USE_TOPDOWN:
            one_step_top_down();
            break;
        case USE_BOTTOMUP:
            one_step_bottom_up();
            break;
        case USE_GPU:
            one_step_gpu_from_cpu();
            break;
        default:
            assert(0);
        }

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
        if (rank == root_owner) s_stop = MPI_Wtime();
#endif

#ifdef SHOWTIMER
        if (rank == root_owner)
            PRINTLN("[TIMER] time for comm : %.6lfs", s_stop - s_start);
#endif

#ifdef SHOWTIMER
        if (rank == root_owner) {
            t_stop = MPI_Wtime();
            t_total = t_stop - t_start;
        }
#endif

#ifdef SHOWTIMER
    if (rank == root_owner)
        PRINTLN("[TIMER] time for level: %.6lfs", t_total);
#endif

        // TODO
        // try
        // if (!frontier_have_more_gpu_from_cpu())
        // but this seems fast enough ~ 0.000001s (each level ~ 0.000248s)
        if (!frontier_have_more())
            break;
    }
    // pred_from_gpu(); // ...
    wrap_up();
}

