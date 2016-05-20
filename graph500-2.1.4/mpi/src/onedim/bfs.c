
#include "bfs.h"

#include <stdlib.h>

#include "constants.h"
#include "print.h"

extern oned_csr_graph g;
int64_t* pred;

#define SET_LOCAL(v, a) do {(a)[VERTEX_LOCAL((v)) / LONG_BITS] |= (1UL << (VERTEX_LOCAL((v)) % LONG_BITS));} while (0)
#define TEST_LOCAL(v, a) (((a)[VERTEX_LOCAL((v)) / LONG_BITS] & (1UL << (VERTEX_LOCAL((v)) % LONG_BITS))) != 0)

#define SET_GLOBAL(v, a) do {(a)[(v) / LONG_BITS] |= (1UL << ((v) % LONG_BITS));} while (0)
#define TEST_GLOBAL(v, a) (((a)[(v) / LONG_BITS] & (1UL << ((v) % LONG_BITS))) != 0)

void show_pred() {
    int i;
    PRINT("rank %02d: index:", rank)
    for (i = 0; i < g.nlocalverts; i++) {
        PRINT(" %2d", VERTEX_TO_GLOBAL(rank, i))
    }
    PRINTLN("")
    PRINT("rank %02d: pred :", rank)
    for (i = 0; i < g.nlocalverts; i++) {
        PRINT(" %2d", pred[i])
    }
    PRINTLN("")
}

const int64_t size_parent_each = (1 << (18)); // worst case: nlocalvert
int64_t size_parent_total;
int64_t size_counter;
int* send_count; // store the number of parent sending out
int* sdispls;
int* receive_count;
int* rdispls;
int64_t* parent_cur; // [parent sending to rank-0 (parent1, count1, parent2, count2) : parent sending to rank-1 : ... ]
int64_t* parent_next; // [parent sending to rank-0 (parent1, count1, parent2, count2) : parent sending to rank-1 : ... ]


int have_more() {
    MPI_Barrier(MPI_COMM_WORLD);
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
    for (i = 0; i < size_counter; i++) {
        PRINT(" [%d]%d", i, (int)send_count[i])
    }
    PRINTLN("")
    PRINT("rank %02d: recv cnt:", rank)
    for (i = 0; i < size_counter; i++) {
        PRINT(" [%d]%d", i, (int)receive_count[i])
    }
    PRINTLN("")
}

void show_displs() {
    PRINT("rank %02d: sdispls :", rank)
    int i;
    for (i = 0; i < size_counter; i++) {
        PRINT(" [%d]%d", i, (int)sdispls[i])
    }
    PRINTLN("")
    PRINT("rank %02d: rdispls:", rank)
    for (i = 0; i < size_counter; i++) {
        PRINT(" [%d]%d", i, (int)rdispls[i])
    }
    PRINTLN("")
}

void show_parent() {
    PRINT("rank %02d: par cur :", rank)
    int i;
    for (i = 0; i < size_parent_total; i++) {
        if (parent_cur[i] != -1)
            PRINT(" [%d]%d>%d", i, parent_cur[i], parent_cur[i + 1])
        i++;
    }
    PRINTLN("")
    PRINT("rank %02d: par next:", rank)
    for (i = 0; i < size_parent_total; i++) {
        if (parent_next[i] != -1)
            PRINT(" [%d]%d>%d", i, parent_next[i], parent_next[i + 1])
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

void bottom_up() {

}

void one_step() {
#ifdef SHOWDEBUG
    REACH_HERE
    if (rank == 0) {
        show_parent();
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else {
        MPI_Barrier(MPI_COMM_WORLD);
        show_parent();
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif

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
            int j;
            for (j = g.rowstarts[to_local]; j < g.rowstarts[to_local + 1]; j++) {
                int64_t next_to = g.column[j];
                add_parent(to, next_to);
            }
        }
    }
}

void init() {
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

void bfs(oned_csr_graph* gg, int64_t root, int64_t* predpred) {
    init();

    pred = predpred;
    memset(pred, -1, g.nlocalverts * sizeof(int64_t));

	if (rank == VERTEX_OWNER(root)) {
#ifdef SHOWDEBUG
        PRINTLN("rank %02d: root: %d", rank, (int)root)
#endif
        parent_cur[0] = parent_cur[1] = root;
        one_step();
#ifdef SHOWDEBUG
        show_parent();
        show_counter();
        show_pred();
#endif
    }

    while (1) {
        if (!have_more())
            break;

        sync();
        one_step();
#ifdef SHOWDEBUG
        PRINTLN("rank %02d: one more level", rank)
        show_parent();
        show_counter();
        show_pred();
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
}

