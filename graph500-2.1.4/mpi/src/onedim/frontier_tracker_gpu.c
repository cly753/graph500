
#include "frontier_tracker.h"

#include "bfs.h"

extern int64_t *frontier_g; // point to gpu memory
extern int64_t *frontier_next_g; // point to gpu memory

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