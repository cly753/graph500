#include "frontier_tracker.h"

#include <string.h>

#include "oned_csr.h"

#include "bfs.h"
#include "print.h"

int sync_times;
int64_t cur_idx;
int size_frontier_idx = 1000;
int64_t *frontier_idx;

int size_receive;
int64_t *frontier_idx_receive;
int thres = 1;

void init_frontier() {
    if (frontier == NULL)
        frontier = xmalloc(global_long_nb);
    if (frontier_next == NULL)
        frontier_next = xmalloc(global_long_nb);
    if (frontier_idx == NULL)
        frontier_idx = xmalloc(size_frontier_idx * sizeof(int64_t));

    memset(frontier, 0, global_long_nb);
    memset(frontier_next, 0, global_long_nb);
    memset(frontier_idx, -1, size_frontier_idx * sizeof(int64_t));

    sync_times = 0;
    size_receive = size_frontier_idx * size;
    if (frontier_idx_receive == NULL)
        frontier_idx_receive = xmalloc(size_receive * sizeof(int64_t));

    // PRINTLN_RANK("size_receive = %d", size_receive)
}

void sync_frontier() {
    if (sync_times < thres) {
        // REACH_HERE_RANK

        if (cur_idx > size_frontier_idx) {
            PRINTLN_RANK("!!! cur_idx = %d, size_frontier_idx = %d !!!", cur_idx, size_frontier_idx)
        }
        // int i;
        // PRINT_RANK("frontier_idx:")
        // for (i = 0; i < size_frontier_idx; i++) {
        //     PRINT(" %d", frontier_idx[i])
        // }
        // PRINTLN("")

        MPI_Allgather(
            frontier_idx, // void* send_data,
            size_frontier_idx, // int send_count,
            MPI_LONG_LONG, // MPI_Datatype send_datatype,
            frontier_idx_receive, // void* recv_data,
            size_frontier_idx, // int recv_count,
            MPI_LONG_LONG, // MPI_Datatype recv_datatype,
            MPI_COMM_WORLD); // MPI_Comm communicator)
        
        // PRINT_RANK("frontier_idx_receive:")
        // for (i = 0; i < size_receive; i++) {
        //     PRINT(" %d", frontier_idx_receive[i])
        // }
        // PRINTLN("")

        sync_times++;
        cur_idx = 0;
        memset(frontier_idx, -1, size_frontier_idx * sizeof(int64_t));
        int i;
        for (i = 0; i < size_receive; i++) {
            if (frontier_idx_receive[i] != -1) {
                SET_GLOBAL(frontier_idx_receive[i], frontier);
            }
        }
        // REACH_HERE_RANK
    }
    else {
        // REACH_HERE_RANK
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

        // REACH_HERE_RANK
    }
}

int frontier_have_more() {
#ifdef USE_OPENMP
    omp_set_num_threads(2);
#endif

    int yes = 0;
    int i;

#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (i = 0; i < global_long_n; i++) {
        if (frontier[i])
#ifdef USE_OPENMP
            yes = 1;
#else
            return 1;
#endif  
    }
    return yes;
}

void add_frontier_next(int64_t to) {
    if (sync_times < thres) {
        frontier_idx[cur_idx++] = to;
    }
    else {
        SET_GLOBAL(to, frontier_next);
    }
}