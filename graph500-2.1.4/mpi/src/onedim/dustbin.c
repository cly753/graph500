

//extern int64_t local_long_n;
//extern int64_t global_long_n;
//extern int64_t global_long_n;
//extern int64_t global_long_nb;
//extern unsigned long *g_cur;
//extern unsigned long *g_next;

//void sync() {
////    int MPI_Allreduce(void *sendbuf, void *recvbuf, int count,
////                      MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
//
////    MPI_Allreduce(g_next, g_cur, global_long_n, MPI_LONG, MPI_BOR, MPI_COMM_WORLD);
////    memset(g_next, 0, sizeof(global_long_nb));
//}

//void one_step() {
//    int i;
//    for (i = 0; i < g.nlocalverts; i++) {
//        int idx_global = VERTEX_TO_GLOBAL(rank, i);
//        if (!TEST_LOCAL(i, visited)) {
//            if (TEST_GLOBAL(idx_global, g_cur)) {
//                SET_LOCAL(i, visited);
//                int j;
//                PRINTLN("rank %02d: index: %d", rank, idx_global)
//                for (j = g.rowstarts[i]; j < g.rowstarts[i + 1]; j++) {
//                    int64_t to = g.column[j];
//                    PRINTLN("rank %02d: %d -> %"PRId64"", rank, idx_global, to)
////                     SET_GLOBAL(to, g_next);
//
//                    if (!TEST_LOCAL(VERTEX_LOCAL(to), visited)) {
//                        pred[VERTEX_LOCAL(to)] = idx_global;
//                    }
//                }
//            }
//
//        }
//    }
//}



//void show_all() {
////    PRINTLN("rank %02d: g_cur:", rank)
////    show_global(g_cur);
////    PRINTLN("rank %02d: g_next:", rank)
////    show_global(g_next);
////    PRINTLN("rank %02d: visited:", rank)
////    show_local(visited);
////    show_pred();
//}
