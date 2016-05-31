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

void show_frontier() {
    PRINT_RANK("frontier_g     : ")
    int64_t *frontier_g_copy = get_frontier_g();
    show_global(frontier_g_copy);
    free(frontier_g_copy);
    PRINT_RANK("frontier_next_g: ")
    int64_t *frontier_next_g_copy = get_frontier_next_g();
    show_global(frontier_next_g_copy);
    free(frontier_next_g_copy);
}

void show_pred() {
    // int i;
    // PRINT_RANK("cpu index:")
    // for (i = 0; i < g.nlocalverts; i++) {
    //     PRINT(" %2d", (int) VERTEX_TO_GLOBAL(rank, i))
    // }
    // PRINTLN("")
    // PRINT_RANK("cpu pred :")
    // for (i = 0; i < g.nlocalverts; i++) {
    //     PRINT(" %2d", (int) pred[i])
    // }
    // PRINTLN("")
    
    int i;
    PRINT_RANK("cpu index:")
    for (i = 0; i < g.nlocalverts; i++) {
        PRINT(" %"PRId64"", VERTEX_TO_GLOBAL(rank, i))
    }
    PRINTLN("")
    PRINT_RANK("cpu pred :")
    for (i = 0; i < g.nlocalverts; i++) {
        PRINT(" %"PRId64"", pred[i])
    }
    PRINTLN("")
}

extern int64_t *in_edge_start;
extern int64_t *in_edge_to;

void init() {
    memset(pred, -1, g.nlocalverts * sizeof(int64_t));
}

void wrap_up() {
    // int visit_node = 0;
    // int i;
    // for (i = 0; i < g.nlocalverts; i++) {
    //     if (pred[i] != -1) {
    //         visit_node += g.rowstarts[i + 1] - g.rowstarts[i];
    //     }
    // }

    // PRINTLN_RANK("visit_node=%d", visit_node)
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

void show_frontier_ordered() {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        show_frontier();
        MPI_Barrier(MPI_COMM_WORLD);
        PRINTLN_RANK("frontier_have_more_gpu: %d", frontier_have_more_gpu())
    }
    else {
        MPI_Barrier(MPI_COMM_WORLD);
        show_frontier();
        PRINTLN_RANK("frontier_have_more_gpu: %d", frontier_have_more_gpu())
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void check_cuda_aware_support() {
    printf("Compile time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    have_cuda_aware_support = 1;
    printf("This MPI library has CUDA-aware support.\n", MPIX_CUDA_AWARE_SUPPORT);
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
    have_cuda_aware_support = 0;
    printf("This MPI library does not have CUDA-aware support.\n");
#else
    have_cuda_aware_support = 0;
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */
 
    printf("Run time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    if (1 == MPIX_Query_cuda_support()) {
        have_cuda_aware_support = 1;
        printf("This MPI library has CUDA-aware support.\n");
    } else {
        have_cuda_aware_support = 0;
        printf("This MPI library does not have CUDA-aware support.\n");
    }
#else /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
    have_cuda_aware_support = 0;
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */
}

void bfs_gpu_cuda_ompi(int64_t root) {
    if (!have_cuda_aware_support)
        init_frontier();

    if (rank == root_owner) {
#ifdef SHOWDEBUG
        PRINTLN("rank %02d: root: %d", rank, (int)root)
#endif
    }

#ifdef SHOWTIMER
    double t_start = 0;
    double t_stop = 0;
    double t_total = 0;
#endif

    init_pred_gpu(root, rank == root_owner);
    set_frontier_gpu(root);
    while (1) {
#ifdef SHOWTIMER
        if (rank == root_owner)
            t_start = MPI_Wtime();
#endif

        one_step_bottom_up_gpu();

        if (have_cuda_aware_support) {
            sync_frontier_gpu();
        }
        else {

            read_frontier_next_g();

            sync_frontier_work_around();

            save_frontier_g();

        }

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
    
        if (!frontier_have_more_gpu())
            break;
    }

    pred_from_gpu();
}

void bfs(oned_csr_graph *gg, int64_t root, int64_t *predpred) {
    pred = predpred;
    root_owner = VERTEX_OWNER(root);

    bfs_gpu_cuda_ompi(root);

    wrap_up();
}

