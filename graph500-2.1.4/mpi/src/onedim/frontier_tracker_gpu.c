
#include "frontier_tracker.h"

#include "bfs.h"


extern int64_t *frontier_g; // point to gpu memory
extern int64_t *frontier_next_g; // point to gpu memory

// use this and read_frontier_next_g() and save_frontier_g()
// if cuda-aware support is not available
void init_frontier() {
    if (frontier == NULL)
        frontier = xmalloc(global_long_nb);
    if (frontier_next == NULL)
        frontier_next = xmalloc(global_long_nb);

    // memset(frontier, 0, global_long_nb);
    // memset(frontier_next, 0, global_long_nb);
}

// use this
// if cuda-aware support is available
// it pass pointer to GPU memory to MPI_Allreduce
// the traffic will go through PCI-e to other GPU
void sync_frontier_gpu() {
    // allreduce to broadcast frontier_g
    MPI_Allreduce(
            frontier_next_g, // void* send_data
            frontier_g, // void* recv_data
            global_long_n, // int count
            MPI_LONG, // MPI_Datatype datatype
            MPI_BOR, // MPI_Op op
            MPI_COMM_WORLD // MPI_Comm communicator
    );
}

// use this and read_frontier_next_g() and save_frontier_g()
// if cuda-aware support is not available
void sync_frontier_work_around() {
    // allreduce to broadcast frontier_g
    MPI_Allreduce(
            frontier_next, // void* send_data
            frontier, // void* recv_data
            global_long_n, // int count
            MPI_LONG, // MPI_Datatype datatype
            MPI_BOR, // MPI_Op op
            MPI_COMM_WORLD // MPI_Comm communicator
    );
}