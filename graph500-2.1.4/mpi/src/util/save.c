
#include "save.h"

#include <mpi.h>
#include <unistd.h>


#define REACH_HERE {;}
#define PRINT(s, ...) {;}
#define PRINTLN(s, ...) {;}
#include "print.h"

#define GRAPH_PREFIX "graph_data_"
#define ROOT_PREFIX "root_data_"

void show_error(int e) {
	if (e != MPI_SUCCESS) {
	   char s[1000];
	   int len;

	   MPI_Error_string(e, s, &len);
	   fprintf(stderr, "error: %s\n", s);
	}
}

const char* get_graph_file_name(int scale, int degree) {
	char cwd[1024];
   	if (getcwd(cwd, sizeof(cwd)) != NULL)
    	fprintf(stdout, "Current working dir: %s\n", cwd);

	return NULL;
    return "/graph500-2.1.4/mpi/data/graph.data";
}

const char* get_root_file_name(int num_bfs_roots) {
	return "root.data";
	return "/graph500-2.1.4/mpi/data/root.data";
}

bool save_tuple_graph(tuple_graph* tg, const char* filename) {
	return false;
}

bool load_tuple_graph(tuple_graph* tg, const char* filename) {
    return false;
}

bool save_bfs_root(int64_t *bfs_roots, int num_bfs_roots, const char* filename) {
	// int MPI_File_open (MPI_Comm comm, char *filename, int amode, MPI_Info info, MPI_File *fh);
	if (rank == 0) {
		int e;

		MPI_File mf;
		PRINTLN("filename: %s", filename)
		MPI_File_set_errhandler(MPI_FILE_NULL, MPI_ERRORS_ARE_FATAL);
		e = MPI_File_open(MPI_COMM_SELF, (char *) filename,
	                      MPI_MODE_RDWR | MPI_MODE_CREATE | MPI_MODE_EXCL,
	                      MPI_INFO_NULL, &mf);
		e = MPI_File_set_size(mf, 1 * sizeof(MPI_INT));
		MPI_Offset fsize;
		e = MPI_File_get_size(mf, &fsize);
		fprintf(stderr, "file size: %d\n", fsize);
		e = MPI_File_set_view(mf, 0, MPI_INT, MPI_INT, "native", MPI_INFO_NULL);
		
		int val = 13;
		// MPI_Status status;
		e = MPI_File_write(&mf, &val, 1, MPI_INT, MPI_STATUS_IGNORE);

		// MPI_File_Close(&mf);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	return true;
}

bool load_bfs_root(int64_t *bfs_roots, int num_bfs_roots, const char* filename) {
	return false;
}

bool load(tuple_graph* tg, int scale, int degree, int64_t *bfs_roots, int num_bfs_roots) {
	return load_bfs_root(bfs_roots, num_bfs_roots, get_root_file_name(num_bfs_roots))
	&& load_tuple_graph(tg, get_graph_file_name(scale, degree));
}

bool save(tuple_graph* tg, int scale, int degree, int64_t *bfs_roots, int num_bfs_roots) {
	return save_bfs_root(bfs_roots, num_bfs_roots, get_root_file_name(num_bfs_roots))
	&& save_tuple_graph(tg, get_graph_file_name(scale, degree));
}


