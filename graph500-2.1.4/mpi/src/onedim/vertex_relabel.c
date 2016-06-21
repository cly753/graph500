
#include "vertex_relabel.h"


int64_t max_index = -1;

int64_t exist_long_n;
int64_t *exist;
int64_t *zero_count_pre;

void show_exist() {
    PRINTLN_RANK("exist:")
    {
        int i;
        char s[65];
        for (i = 0; i < exist_long_n; i++) {
            print_binary_long(exist[i], s);
            PRINT("%s ", s);
        }
        PRINTLN("")    
    }
}

#define MAX(x, y) ((x) > (y) ? (x) : (y))

int64_t get_new_index(int64_t old_index) {
    int64_t pre = zero_count_pre[old_index / LONG_BITS];
    int i;
    for (i = old_index / LONG_BITS * LONG_BITS; i < old_index; i++) {
        if (!TEST_GLOBAL(i, exist))
            pre++;
    }
    int64_t new_index = old_index - pre;
#ifdef SHOWDEBUG
    // PRINTLN_RANK("old_index = %d, pre = %d -> new_index = %d", (int)old_index, (int)pre, (int)new_index)
    mpi_assert(old_index < 128);
    mpi_assert(new_index < 128);
#endif
    return new_index;
}

int64_t get_old_index(int64_t new_index) {
#ifdef SHOWDEBUG
    // PRINTLN_RANK("get_old_index new_index = %d", (int)new_index)
#endif    
    int one_need = new_index + 1;
    int l = 0;
    int r = exist_long_n;
    while (l + 1 < r) {
        int m = (l + r) / 2;
        if (m * LONG_BITS - zero_count_pre[m] >= one_need) {
            r = m;
        }
        else {
            l = m;
        }
    }
    int old_index = l * LONG_BITS;
    int more = one_need - (l * LONG_BITS - zero_count_pre[l]);
#ifdef SHOWDEBUG
    // PRINTLN_RANK("new_index = %d, l = %d, old_index = %d, more = %d", new_index, l, old_index, more)
#endif
    if (TEST_GLOBAL(old_index, exist))
        more--;
    while (more) {
        old_index++;
        if (TEST_GLOBAL(old_index, exist))
            more--;
    }
    while (!TEST_GLOBAL(old_index, exist)) {
        old_index++;
    }
#ifdef SHOWDEBUG
    // PRINTLN_RANK("new_index = %d, old_index = %d", new_index, old_index)
    mpi_assert(new_index < 128);
    mpi_assert(old_index < 128);
#endif    
    return old_index;
}

int been_relabelled(int64_t old_index) {
	return TEST_GLOBAL(old_index, exist);
}

// assume not define GENERATOR_USE_PACKED_EDGE_TYPE
// see "../../../generator/graph_generator.h"
int64_t filter_zero_degree(const tuple_graph* const tg) {
#ifdef USE_OPENMP
    omp_set_num_threads(24);
#endif

#ifdef GENERATOR_USE_PACKED_EDGE_TYPE
    PRINTLN_RANK("GENERATOR_USE_PACKED_EDGE_TYPE yes filter_zero_degree skip")
    mpi_assert(false);
#endif

    if (rank != 0)
        return -1;

    int i;
    for (i = 0; i < tg->edgememory_size; i++) {
        int64_t v0 = get_v0_from_edge(tg->edgememory + i);
        int64_t v1 = get_v1_from_edge(tg->edgememory + i);
        max_index = MAX(max_index, v0);
        max_index = MAX(max_index, v1);
    }

    exist_long_n = (max_index + 1 + LONG_BITS - 1) / LONG_BITS;
    exist = xmalloc(exist_long_n * sizeof(int64_t));
    memset(exist, 0, exist_long_n * sizeof(int64_t));
    zero_count_pre = xmalloc(exist_long_n * sizeof(int64_t));
    for (i = 0; i < tg->edgememory_size; i++) {
        int64_t v0 = get_v0_from_edge(tg->edgememory + i);
        int64_t v1 = get_v1_from_edge(tg->edgememory + i);
        SET_GLOBAL(v0, exist);
        SET_GLOBAL(v1, exist);
    }

#ifdef SHOWDEBUG
    show_exist();
#endif
    
    int64_t accu = 0;
    for (i = 0; i <= max_index; i++) {
        if (i % LONG_BITS == 0) {
            zero_count_pre[i / LONG_BITS] = accu;
        }
        if (!TEST_GLOBAL(i, exist)) {
#ifdef SHOWDEBUG            
            PRINTLN_RANK("%d not exist", (int)i)
#endif            
            accu++;
        }
        else {
#ifdef SHOWDEBUG
            PRINTLN_RANK("%d     exist", (int)i)
#endif
        }
    }

#ifdef SHOWDEBUG
    PRINTLN_RANK("zero degree accu: %"PRId64"", accu)
    for (i = 0; i <= max_index; i++) {
        if (TEST_GLOBAL(i, exist)) {
            int64_t new_idx = get_new_index(i);
            int64_t old_idx = get_old_index(new_idx);
            PRINTLN_RANK("old %"PRId64" -> new %"PRId64, i, new_idx);
            mpi_assert(i == old_idx);
        }
    }
#endif


#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (i = 0; i < tg->edgememory_size; i++) {
        int64_t v0 = get_v0_from_edge(tg->edgememory + i);
        int64_t v1 = get_v1_from_edge(tg->edgememory + i);
        int64_t v0_new = get_new_index(v0);
        int64_t v1_new = get_new_index(v1);
        write_edge(tg->edgememory + i, v0_new, v1_new);
    }

    return max_index + 1 - accu;
}

void broadcast_filter_zero_degree_result() {
    MPI_Bcast(
        &exist_long_n, // void* data,
        1, // int count,
        MPI_LONG_LONG, // MPI_Datatype datatype,
        0, // int root,
        MPI_COMM_WORLD); // MPI_Comm communicator)
    if (rank != 0) {
        exist = xmalloc(exist_long_n * sizeof(int64_t));
        zero_count_pre = xmalloc(exist_long_n * sizeof(int64_t));
    }

    MPI_Bcast(
        exist, // void* data,
        exist_long_n, // int count,
        MPI_LONG_LONG, // MPI_Datatype datatype,
        0, // int root,
        MPI_COMM_WORLD); // MPI_Comm communicator)
    MPI_Bcast(
        zero_count_pre, // void* data,
        exist_long_n, // int count,
        MPI_LONG_LONG, // MPI_Datatype datatype,
        0, // int root,
        MPI_COMM_WORLD); // MPI_Comm communicator)
}

int *remap_receive_count; // N vertex, should at this node, but remapped away to i-th node
int64_t *remap_receive_buf;
int* remap_receive_buf_start;
int remap_receive_total;
int *remap_send_count; // N vertex, should at i-th node, but remapped to this node
int64_t *remap_send_buf;
int* remap_send_buf_start;
int* remap_send_buf_current;
int remap_send_total;

int *is_old_owner;
int64_t *pred_to_send; // [old idx, new idx local, old idx, new idx local, ...]

void calculate_pred_to_send() {
	memcpy(remap_send_buf_current, remap_send_buf_start, size * sizeof(int));
    int i;
    for (i = non_zero_degree_count - 1; i >= 0; i--) {
        int64_t new_idx = VERTEX_TO_GLOBAL(rank, i);
        int64_t child_old_index = get_old_index(new_idx);
        int owner = VERTEX_OWNER(child_old_index);
        // PRINTLN_RANK("(%d) new idx = %d, child_old_index = %d, owner = %d",
        	// i, new_idx, child_old_index, owner)
        if (owner == rank) {
        	is_old_owner[i] = 1;
        }
        else {
        	is_old_owner[i] = 0;

        	pred_to_send[remap_send_buf_current[owner]++] = child_old_index;
        	pred_to_send[remap_send_buf_current[owner]++] = i;
        }
    }
}

void calculate_remapped_count() {
#ifdef SHOWDEBUG
    PRINTLN_RANK("non_zero_degree_count %d", non_zero_degree_count)
#endif

    remap_receive_count = xmalloc(size * sizeof(int));
    memset(remap_receive_count, 0, size * sizeof(int));
    remap_send_count = xmalloc(size * sizeof(int));
    memset(remap_send_count, 0, size * sizeof(int));
    
    int i;
    for (i = 0; i < non_zero_degree_count; i++) {
        int64_t global_i = VERTEX_TO_GLOBAL(rank, i);
        int current_owner = VERTEX_OWNER(global_i);
        int64_t old_index = get_old_index(global_i);
        int real_owner = VERTEX_OWNER(old_index);
#ifdef SHOWDEBUG
        PRINTLN_RANK("new index %d <- old index %d (current_owner %d real_owner %d)", 
            global_i, old_index, current_owner, real_owner)
#endif        
        if (current_owner == rank && real_owner != rank) {
            remap_send_count[real_owner] += 2; // pred[x]=y -> (x,y) <- x2
#ifdef SHOWDEBUG            
            PRINTLN_RANK("not mine")
#endif            
        }
    }
    remap_send_total = 0;
    for (i = 0; i < size; i++)
        remap_send_total += remap_send_count[i];
    
    remap_send_buf = xmalloc(remap_send_total * sizeof(int64_t));
    memset(remap_send_buf, 0, remap_send_total * sizeof(int64_t));
    
    remap_send_buf_start = xmalloc(size * sizeof(int));
    remap_send_buf_start[0] = 0;
    for (i = 1; i < size; i++)
        remap_send_buf_start[i] = remap_send_buf_start[i - 1] + remap_send_count[i - 1];
    remap_send_buf_current = xmalloc(size * sizeof(int));
    memset(remap_send_buf_current, 0, size * sizeof(int));
    MPI_Alltoall(remap_send_count, // const void *sendbuf
                 1, // int sendcount
                 MPI_INT, // MPI_Datatype sendtype
                 remap_receive_count, // void *recvbuf
                 1, // int recvcount
                 MPI_INT, // MPI_Datatype recvtype
                 MPI_COMM_WORLD); // MPI_Comm comm

#ifdef SHOWDEBUG
    for (i = 0; i < size; i++) {
        PRINTLN_RANK("to %d, send %d, receive %d", i, remap_send_count[i], remap_receive_count[i])
    }
#endif    

    remap_receive_buf_start = xmalloc(size * sizeof(int));
    remap_receive_buf_start[0] = 0;
    for (i = 1; i < size; i++)
        remap_receive_buf_start[i] = remap_receive_buf_start[i - 1] + remap_receive_count[i - 1];
    remap_receive_total = 0;
    for (i = 0; i < size; i++)
        remap_receive_total += remap_receive_count[i];
    remap_receive_buf = xmalloc(remap_receive_total * sizeof(int64_t));
    memset(remap_receive_buf, 0, remap_receive_total * sizeof(int64_t));

    pred_to_send = xmalloc(remap_send_total * sizeof(int64_t));
    memset(pred_to_send, 0, remap_send_total * sizeof(int64_t));
    is_old_owner = xmalloc(non_zero_degree_count * sizeof(int));
    calculate_pred_to_send();
}

void recover_index(int64_t *pred) {
#ifdef USE_OPENMP
    omp_set_num_threads(24);
#endif

    memcpy(remap_send_buf_current, remap_send_buf_start, size * sizeof(int));
    int i;
    for (i = 0; i < remap_send_total; ) {
    	int64_t child_old_index = pred_to_send[i];
    	remap_send_buf[i] = child_old_index;
    	i++;
       	int64_t new_index_local = pred_to_send[i];
       	int64_t parent_index_global = pred[new_index_local];
       	pred[new_index_local] = -1;
       	if (parent_index_global != -1)
       		parent_index_global = get_old_index(parent_index_global);
       	remap_send_buf[i] = parent_index_global;
    	i++;
    }

#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (i = non_zero_degree_count - 1; i >= 0; i--) {
    	if (!is_old_owner[i])
    		continue ;

#ifdef SHOWDEBUG
        show_exist();
#endif
        int64_t new_idx = VERTEX_TO_GLOBAL(rank, i);
        int64_t child_old_index = get_old_index(new_idx);
#ifdef SHOWDEBUG
        PRINTLN_RANK("recovering idx %d new %d (old %d) ANOTHER WAY %d", i, (int)new_idx, (int)child_old_index, (int)get_old_index(VERTEX_TO_GLOBAL(rank, i)))
#endif
        int64_t parent_old_index = pred[i];
        if (parent_old_index != -1)
            parent_old_index = get_old_index(pred[i]);

        int64_t old_local = VERTEX_LOCAL(child_old_index);
        pred[old_local] = parent_old_index;
#ifdef SHOWDEBUG
        PRINTLN_RANK("put pred[%d]=%d to pred[%d]=%d", 
            (int)VERTEX_TO_GLOBAL(rank, i), (int)pred[i], 
            (int)get_old_index(VERTEX_TO_GLOBAL(rank, i)), (int)parent_old_index)
#endif
        if (old_local != i)
            pred[i] = -1;
    }
    
    MPI_Alltoallv(remap_send_buf, // const void *sendbuf
		remap_send_count, // const int sendcounts[]
		remap_send_buf_start, // const int sdispls[]
		MPI_LONG, // MPI_Datatype sendtype
		remap_receive_buf, // void *recvbuf
		remap_receive_count, // const int recvcounts[]
		remap_receive_buf_start, // const int rdispls[]
		MPI_LONG, // MPI_Datatype recvtype
		MPI_COMM_WORLD); // MPI_Comm comm
    
    for (i = 0; i < remap_receive_total; ) {
        int64_t child_old_index = remap_receive_buf[i];
        i++;
        int64_t parent_old_index = remap_receive_buf[i];
        i++;
        pred[VERTEX_LOCAL(child_old_index)] = parent_old_index;
#ifdef SHOWDEBUG
        PRINTLN_RANK("I receive: child_old_index %d, parent_old_index %d", 
            (int)child_old_index, (int)parent_old_index)
#endif
    }
}
