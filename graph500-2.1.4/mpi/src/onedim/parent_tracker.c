#include "parent_tracker.h"

#include <string.h>

int have_more() {
    int have_more_local[2] = {0, 0};
    int i;
    for (i = 0; i < size; i++)
        have_more_local[0] |= send_count[i];
#ifdef SHOWDEBUG
    PRINTLN("rank %02d: have[0]: %d", rank, have_more_local[0])
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    MPI_Allreduce(&have_more_local[0], // void* send_data
                  &have_more_local[1], // void* recv_data
                  1, // int count
                  MPI_INT, // MPI_Datatype datatype
                  MPI_BOR, // MPI_Op op
                  MPI_COMM_WORLD); // MPI_Comm communicator
#ifdef SHOWDEBUG
    PRINTLN("rank %02d: have[1]: %d", rank, have_more_local[1])
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    return have_more_local[1];
}

void show_counter() {
    PRINT("rank %02d: send cnt:", rank)
    int i;
    for (i = 0; i < size_counter; i++) PRINT(" [%d]%d", i, (int) send_count[i])
    PRINTLN("")
    PRINT("rank %02d: recv cnt:", rank)
    for (i = 0; i < size_counter; i++) PRINT(" [%d]%d", i, (int) receive_count[i])
    PRINTLN("")
}

void show_displs() {
    PRINT("rank %02d: sdispls :", rank)
    int i;
    for (i = 0; i < size_counter; i++) PRINT(" [%d]%d", i, (int) sdispls[i])
    PRINTLN("")
    PRINT("rank %02d: rdispls:", rank)
    for (i = 0; i < size_counter; i++) PRINT(" [%d]%d", i, (int) rdispls[i])
    PRINTLN("")
}

void show_parent() {
    PRINT("rank %02d: par cur :", rank)
    int i;
    for (i = 0; i < size_parent_total; i++) {
        if (parent_cur[i] != -1) PRINT(" [%d]%d>%d", i, parent_cur[i], parent_cur[i + 1])
        i++;
    }
    PRINTLN("")
    PRINT("rank %02d: par next:", rank)
    for (i = 0; i < size_parent_total; i++) {
        if (parent_next[i] != -1) PRINT(" [%d]%d>%d", i, parent_next[i], parent_next[i + 1])
        i++;
    }
    PRINTLN("")
}

void add_parent(int64_t from, int64_t to) {
    int owner = VERTEX_OWNER(to);
    int64_t idx = owner * size_parent_each + send_count[owner];
#ifdef SHOWDEBUG
    PRINTLN("rank %02d: add parent from %2d to %2d at idx: %d", rank, from, to, (int)idx)
#endif
    parent_next[idx] = from;
    idx++;
    parent_next[idx] = to;
    send_count[owner] += 2;
}

void sync_counter() {
#ifdef SHOWDEBUG
    PRINTLN("rank %02d: sync_counter", rank)
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        show_counter();
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else {
        MPI_Barrier(MPI_COMM_WORLD);
        show_counter();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    PRINTLN("rank %02d: syncing", rank)
#endif
    MPI_Alltoall(send_count, // const void *sendbuf
                 1, // int sendcount
                 MPI_INT, // MPI_Datatype sendtype
                 receive_count, // void *recvbuf
                 1, // int recvcount
                 MPI_INT, // MPI_Datatype recvtype
                 MPI_COMM_WORLD); // MPI_Comm comm
#ifdef SHOWDEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        show_counter();
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else {
        MPI_Barrier(MPI_COMM_WORLD);
        show_counter();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    PRINTLN("rank %02d: sync_counter finish", rank)
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

void sync() {
    sync_counter();
    memset(parent_cur, -1, size_parent_total * sizeof(int64_t));

#ifdef SHOWDEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    PRINTLN("rank %02d: sync parent", rank)
    if (rank == 0) {
        show_parent();
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else {
        MPI_Barrier(MPI_COMM_WORLD);
        show_parent();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    PRINTLN("rank %02d: syncing parent", rank)
#endif

    int i;
    for (i = 1; i < size_counter; i++)
        sdispls[i] = sdispls[i - 1] + size_parent_each;
    for (i = 1; i < size_counter; i++)
        rdispls[i] = rdispls[i - 1] + receive_count[i - 1];

#ifdef SHOWDEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        show_displs();
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else {
        MPI_Barrier(MPI_COMM_WORLD);
        show_displs();
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    MPI_Alltoallv(parent_next, // const void *sendbuf
                  send_count, // const int sendcounts[]
                  sdispls, // const int sdispls[]
                  MPI_LONG, // MPI_Datatype sendtype
                  parent_cur, // void *recvbuf
                  receive_count, // const int recvcounts[]
                  rdispls, // const int rdispls[]
                  MPI_LONG, // MPI_Datatype recvtype
                  MPI_COMM_WORLD); // MPI_Comm comm

    memset(parent_next, -1, size_parent_total * sizeof(int64_t));
    memset(send_count, 0, size_counter * sizeof(int));

#ifdef SHOWDEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        show_parent();
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else {
        MPI_Barrier(MPI_COMM_WORLD);
        show_parent();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    PRINTLN("rank %02d: sync parent finish", rank)
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

void init_parent_tracker() {
    size_parent_total = size_parent_each * size;
    if (parent_cur == NULL)
        parent_cur = xmalloc(size_parent_total * sizeof(int64_t));
    memset(parent_cur, -1, size_parent_total * sizeof(int64_t));

    if (parent_next == NULL)
        parent_next = xmalloc(size_parent_total * sizeof(int64_t));
    memset(parent_next, -1, size_parent_total * sizeof(int64_t));

    size_counter = size;
    if (send_count == NULL)
        send_count = xmalloc(size_counter * sizeof(int));
    memset(send_count, 0, size_counter * sizeof(int));
    if (receive_count == NULL)
        receive_count = xmalloc(size_counter * sizeof(int));
    memset(receive_count, 0, size_counter * sizeof(int));
    if (sdispls == NULL)
        sdispls = xmalloc(size_counter * sizeof(int));
    memset(sdispls, 0, size_counter * sizeof(int));
    if (rdispls == NULL)
        rdispls = xmalloc(size_counter * sizeof(int));
    memset(rdispls, 0, size_counter * sizeof(int));
}