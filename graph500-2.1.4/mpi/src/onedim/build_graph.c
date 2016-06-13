
#include "build_graph.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stddef.h>
#include "common.h"

#include "constants.h"
#include "print.h"

// typedef struct oned_csr_graph {
//     size_t nlocalverts;
//     int64_t max_nlocalverts;
//     size_t nlocaledges;
//     int lg_nglobalverts;
//     int64_t nglobalverts;
//     size_t *rowstarts;
//     int64_t *column;
//     const tuple_graph* tg; /* Original graph used to build this one */
// } oned_csr_graph;
// 
// typedef struct packed_edge {
//   uint32_t v0_low;
//   uint32_t v1_low;
//   uint32_t high; /* v1 in high half, v0 in low half */
// } packed_edge;
// 
// typedef struct tuple_graph {
//     int data_in_file;
//     /* 1 for file, 0 for memory */
//     packed_edge *restrict edgememory;
//     /* NULL if edges are in file */
//     int64_t edgememory_size;
//     int64_t max_edgememory_size;
//     MPI_File edgefile;
//     /* Or MPI_FILE_NULL if edges are in memory */
//     int64_t nglobaledges; /* Number of edges in graph, in both cases */
// } tuple_graph;

int **all_column_count;
int64_t **all_column_start; // this will be the oned_csr_graph.rowstarts
int64_t **all_column_current;
int *all_column_total;
int64_t** all_column;

extern int64_t max_index; // in vertex_relabel.c

#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

void prepare(tuple_graph *tg, oned_csr_graph *g) {

    int i;
	if (max_index <= 0) {
		for (i = 0; i < tg->edgememory_size; i++) {
	        int64_t v0 = get_v0_from_edge(tg->edgememory + i);
	        int64_t v1 = get_v1_from_edge(tg->edgememory + i);
	        max_index = MAX(max_index, v0);
	        max_index = MAX(max_index, v1);
	    }
	}
	
	int graph_size = 1;
	int lg = 0;
	while (graph_size < max_index)
		graph_size <<= 1, lg++;
	
	g->nlocalverts = graph_size / size;
	g->nglobalverts = graph_size;
	g->lg_nglobalverts = lg;

	g->max_nlocalverts = -1;
	g->nlocaledges = -1;
	g->tg = NULL;

#ifdef SHOWDEBUG
	PRINTLN_RANK("max_index = %d, graph_size = %d, g->nlocalverts = %d", max_index, graph_size, g->nlocalverts)
#endif

	all_column_count = xmalloc(size * sizeof(int*));
	all_column_total = xmalloc(size * sizeof(int));
	memset(all_column_total, 0, size * sizeof(int));
	for (i = 0; i < size; i++) {
		int nb = g->nlocalverts * sizeof(int);
		all_column_count[i] = xmalloc(nb);
		memset(all_column_count[i], 0, nb);
	}

	for (i = 0; i < tg->edgememory_size; i++) {
        int64_t v0 = get_v0_from_edge(tg->edgememory + i);
        int v0_owner = VERTEX_OWNER(v0);
        int v0_local = VERTEX_LOCAL(v0);
        int64_t v1 = get_v1_from_edge(tg->edgememory + i);
		int v1_owner = VERTEX_OWNER(v1);
		int v1_local = VERTEX_LOCAL(v1);

		if (v0 == v1)
			continue;

		all_column_count[v0_owner][v0_local]++; // one edge from v0 to v1
		all_column_total[v0_owner]++;
		all_column_count[v1_owner][v1_local]++; // one edge from v1 to v0
		all_column_total[v1_owner]++;

#ifdef SHOWDEBUG
		PRINTLN_RANK("edge %d(%d, %d) - %d(%d, %d) all_column_total[%d] = %d, all_column_total[%d] = %d", 
			v0, v0_owner, v0_local, v1, v1_owner, v1_local,
			v0_owner, all_column_total[v0_owner],
			v1_owner, all_column_total[v1_owner])
#endif
	}

#ifdef SHOWDEBUG
	for (i = 0; i < size; i++) {
		PRINTLN_RANK("all_column_total[%d] = %d, each:", i, all_column_total[i])
		int j;
		for (j = 0; j < g->nlocalverts; j++) {
			PRINT(" [%d]%d", VERTEX_TO_GLOBAL(i, j), all_column_count[i][j])
		}
		PRINTLN("")
	}
#endif

	all_column_start = xmalloc(size * sizeof(int64_t*));
	all_column_current = xmalloc(size * sizeof(int64_t*));
	for (i = 0; i < size; i++) {
		all_column_start[i] = xmalloc((g->nlocalverts + 1) * sizeof(int64_t));
		all_column_start[i][0] = 0;
		int j;
		for (j = 1; j <= g->nlocalverts; j++) {
			all_column_start[i][j] = all_column_start[i][j - 1] + all_column_count[i][j - 1];
		}
		all_column_current[i] = xmalloc(g->nlocalverts * sizeof(int64_t));
		memcpy(all_column_current[i], all_column_start[i], g->nlocalverts * sizeof(int64_t));
	}

	all_column = xmalloc(size * sizeof(int64_t*));
	for (i = 0; i < size; i++) {
		all_column[i] = xmalloc(all_column_total[i] * sizeof(int64_t));
		memset(all_column[i], 0, all_column_total[i] * sizeof(int64_t));
	}

#ifdef SHOWDEBUG
	for (i = 0; i < size; i++) {
		PRINTLN_RANK("all_column_start[%d] each:", i)
		int j;
		for (j = 0; j <= g->nlocalverts; j++) {
			PRINT(" [%d]%d", VERTEX_TO_GLOBAL(i, j), all_column_start[i][j])
		}
		PRINTLN("")

		PRINTLN_RANK("all_column_current[%d] each:", i)
		for (j = 0; j < g->nlocalverts; j++) {
			PRINT(" [%d]%d", VERTEX_TO_GLOBAL(i, j), all_column_current[i][j])
		}
		PRINTLN("")
	}
#endif

	for (i = 0; i < tg->edgememory_size; i++) {
        int64_t v0 = get_v0_from_edge(tg->edgememory + i);
        int v0_owner = VERTEX_OWNER(v0);
        int v0_local = VERTEX_LOCAL(v0);
        int64_t v1 = get_v1_from_edge(tg->edgememory + i);
		int v1_owner = VERTEX_OWNER(v1);
		int v1_local = VERTEX_LOCAL(v1);

		if (v0 == v1)
			continue;

		all_column[v0_owner][all_column_current[v0_owner][v0_local]] = v1;
		all_column_current[v0_owner][v0_local]++;
		all_column[v1_owner][all_column_current[v1_owner][v1_local]] = v0;
		all_column_current[v1_owner][v1_local]++;
	}

#ifdef SHOWDEBUG
	for (i = 0; i < size; i++) {
		PRINTLN_RANK("all_column_current[%d] each:", i)
		int j;
		for (j = 0; j < g->nlocalverts; j++) {
			PRINT(" [%d]%d", VERTEX_TO_GLOBAL(i, j), all_column_current[i][j])
		}
		PRINTLN("")

		PRINTLN_RANK("all_column[%d] each:", i)
		for (j = 0; j < g->nlocalverts; j++) {
			PRINT("from [%d] to: ", VERTEX_TO_GLOBAL(i, j))
			int k;
			for (k = all_column_start[i][j]; k < all_column_start[i][j + 1]; k++) {
				PRINT(" %d", all_column[i][k])	
			}
			PRINTLN("")
		}
		PRINTLN("")
	}
#endif
}

void broadcast(oned_csr_graph *g) {
	// broadcast g->nlocalverts
	MPI_Bcast(
	    &g->nlocalverts, // void* data,
	    1, // int count,
	    MPI_LONG_LONG, // MPI_Datatype datatype,
	    0, // int root,
	    MPI_COMM_WORLD); // MPI_Comm communicator)

	// braodcast g->nglobalverts
	MPI_Bcast(
	    &g->nglobalverts, // void* data,
	    1, // int count,
	    MPI_LONG_LONG, // MPI_Datatype datatype,
	    0, // int root,
	    MPI_COMM_WORLD); // MPI_Comm communicator)

	// braodcast g->lg_nglobalverts
	MPI_Bcast(
	    &g->lg_nglobalverts, // void* data,
	    1, // int count,
	    MPI_INT, // MPI_Datatype datatype,
	    0, // int root,
	    MPI_COMM_WORLD); // MPI_Comm communicator)

	if (rank != 0) {
		g->max_nlocalverts = -1;
		g->nlocaledges = -1;
		g->lg_nglobalverts = -1;
		g->tg = NULL;

		g->rowstarts = xmalloc((g->nlocalverts + 1) * sizeof(int64_t));
		MPI_Recv(
		    g->rowstarts, // void* data,
		    g->nlocalverts + 1, // int count,
		    MPI_LONG_LONG, // MPI_Datatype datatype,
		    0, // int source,
		    0, // int tag,
		    MPI_COMM_WORLD, // MPI_Comm communicator,
		    MPI_STATUS_IGNORE); // MPI_Status* status)
		
		int total = g->rowstarts[g->nlocalverts];
		g->column = xmalloc(total * sizeof(int64_t));
		MPI_Recv(
		    g->column, // void* data,
		    total, // int count,
		    MPI_LONG_LONG, // MPI_Datatype datatype,
		    0, // int source,
		    0, // int tag,
		    MPI_COMM_WORLD, // MPI_Comm communicator,
		    MPI_STATUS_IGNORE); // MPI_Status* status)
	}
	else {
		g->rowstarts = xmalloc((g->nlocalverts + 1) * sizeof(int64_t));
		memcpy(g->rowstarts, all_column_start[0], (g->nlocalverts + 1) * sizeof(int64_t));
		int total = g->rowstarts[g->nlocalverts];
		g->column = xmalloc(total * sizeof(int64_t));
		memcpy(g->column, all_column[0], total * sizeof(int64_t));

		int i;
		for (i = 1; i < size; i++) {
			MPI_Send(
			    all_column_start[i], // void* data,
			    g->nlocalverts + 1, // int count,
			    MPI_LONG_LONG, // MPI_Datatype datatype,
			    i, // int destination,
			    0, // int tag,
			    MPI_COMM_WORLD); // MPI_Comm communicator)
			MPI_Send(
			    all_column[i], // void* data,
			    all_column_start[i][g->nlocalverts], // int count,
			    MPI_LONG_LONG, // MPI_Datatype datatype,
			    i, // int destination,
			    0, // int tag,
			    MPI_COMM_WORLD); // MPI_Comm communicator)
		}
	}
}

void free_up() {
	int i;
	for (i = 0; i < size; i++) {
		free(all_column_count[i]);
		free(all_column_start[i]);
		free(all_column_current[i]);
		free(all_column[i]);
	}
	free(all_column_count);
	free(all_column_start);
	free(all_column_current);
	free(all_column_total);
	free(all_column);
}

void new_convert_graph_to_oned_csr(tuple_graph *tg, oned_csr_graph *g) {
	assert(sizeof(int64_t) == sizeof(size_t));

	if (rank == 0)
		prepare(tg, g);
	broadcast(g);
	if (rank == 0)
		free_up();
}