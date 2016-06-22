#include "frontier_tracker.h"

#include <string.h>

#include "oned_csr.h"

#include "bfs.h"
#include "print.h"

int sync_times;
int cur_idx;
int size_frontier_idx = 100000;
int64_t *frontier_idx;

int64_t *receive_index;
int size_receive_total;
int thres = 2;

int use_index;

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
}

void sync_frontier() {
    if (sync_times < thres) {
        use_index = 1;

        if (cur_idx > size_frontier_idx) {
            PRINTLN_RANK("!!! cur_idx = %d, size_frontier_idx = %d !!!", cur_idx, size_frontier_idx)
        }

        int *size_receive = xmalloc(size * sizeof(int));
        MPI_Allgather(
            &cur_idx, // void* send_data,
            1, // int send_count,
            MPI_INT, // MPI_Datatype send_datatype,
            size_receive, // void* recv_data,
            1, // int recv_count,
            MPI_INT, // MPI_Datatype recv_datatype,
            MPI_COMM_WORLD); // MPI_Comm communicator)

        size_receive_total = 0;
        int j;
        for (j = 0; j < size; j++) {
            size_receive_total += size_receive[j];
        }
        if (receive_index != NULL)
            free(receive_index);
        receive_index = xmalloc(size_receive_total * sizeof(int64_t));

        int *receive_displs = xmalloc(size * sizeof(int));
        receive_displs[0] = 0;
        for (j = 1; j < size; j++) {
            receive_displs[j] = receive_displs[j - 1] + size_receive[j - 1];
        }
        
        MPI_Allgatherv(
            frontier_idx, // const void *sendbuf, 
            cur_idx, // int sendcount,
            MPI_LONG_LONG, // MPI_Datatype sendtype, 
            receive_index, // void *recvbuf, 
            size_receive, // const int recvcounts[],
            receive_displs, // const int displs[], 
            MPI_LONG_LONG, // MPI_Datatype recvtype, 
            MPI_COMM_WORLD); // MPI_Comm comm)

        // int i;
        // PRINT_RANK("receive_index (%d):", size_receive_total)
        // for (i = 0; i < size_receive_total; i++) {
        //     PRINT(" %d", receive_index[i])
        // }
        // PRINTLN("")

        sync_times++;
        cur_idx = 0;
    }
    else {
        use_index = 0;

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
}

int frontier_have_more() {
    if (use_index) {
        return size_receive_total != 0;
    }

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