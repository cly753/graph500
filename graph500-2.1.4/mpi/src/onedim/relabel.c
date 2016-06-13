
#include "relabel.h"

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
#include "bfs.h"

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

oned_csr_graph *g_ptr;
int64_t *old_label;
int64_t *new_label;
int64_t *pred_temp;

int64_t get_new_label(int64_t old) {
	return new_label[old];
}

int64_t get_old_label(int64_t new_lbl) {
	return old_label[new_lbl];
}

int compare_degree(const void * a, const void * b) {
	int idx_a = *(int*)a;
	int idx_b = *(int*)b;
	int degree_a = g_ptr->rowstarts[idx_a + 1] - g_ptr->rowstarts[idx_a];
	int degree_b = g_ptr->rowstarts[idx_b + 1] - g_ptr->rowstarts[idx_b];
   	return degree_b - degree_a;
}

// can be slow, time not counted
void relabel(oned_csr_graph *g) {
	g_ptr = g;
	PRINTLN_RANK("g_ptr->nlocalverts = %d", g_ptr->nlocalverts)
	old_label = xmalloc(g_ptr->nlocalverts * sizeof(int64_t));
	new_label = xmalloc(g_ptr->nlocalverts * sizeof(int64_t));
	int i;
	for (i = 0; i < g_ptr->nlocalverts; i++) {
		old_label[i] = i;
	}

	// void qsort(void *base, size_t nitems, size_t size, int (*compar)(const void *, const void*))
	qsort(old_label, g_ptr->nlocalverts, sizeof(int64_t), compare_degree);

	for (i = 0; i < g_ptr->nlocalverts; i++) {
		new_label[old_label[i]] = i;
	}

	for (i = 0; i < g_ptr->nlocalverts; i++) {
		PRINTLN_RANK("(%d): %d", i, VERTEX_TO_GLOBAL(rank, get_old_label(i)))
	}

	int64_t *rowstarts_temp = xmalloc((g_ptr->nlocalverts + 1) * sizeof(int64_t));
	int64_t *column_temp = xmalloc(g_ptr->rowstarts[g_ptr->nlocalverts] * sizeof(int64_t));
	int current = 0;
	for (i = 0; i < g_ptr->nlocalverts; i++) {
		// PRINTLN_RANK("current = %d", current)
		rowstarts_temp[i] = current;
		int old = get_old_label(i);
		PRINTLN_RANK("relabel old %d -> new %d", old, i)
		// PRINTLN_RANK("(%d) old = %d [%d, %d)", 
		// 	i, VERTEX_TO_GLOBAL(rank, old), g_ptr->rowstarts[old], g_ptr->rowstarts[old + 1])
		int j;
		for (j = g_ptr->rowstarts[old]; j < g_ptr->rowstarts[old + 1]; j++) {
			// PRINTLN_RANK("j = %d, current = %d", j, current)
			// PRINTLN_RANK("g_ptr->column[%d]", j, g_ptr->column[j])
			column_temp[current] = get_new_label(g_ptr->column[j]);
			current++;
		}
	}

	rowstarts_temp[i] = current;
	mpi_assert(current == g_ptr->rowstarts[g_ptr->nlocalverts]);
	memcpy(g_ptr->rowstarts, rowstarts_temp, (g_ptr->nlocalverts + 1) * sizeof(int64_t));
	memcpy(g_ptr->column, column_temp, g_ptr->rowstarts[g_ptr->nlocalverts] * sizeof(int64_t));
	free(rowstarts_temp);
	free(column_temp);

	pred_temp = xmalloc(g_ptr->nlocalverts * sizeof(int64_t));
}

// need to be fast
void undo_relabel(int64_t* pred) {
	int i;
	for (i = 0; i < g_ptr->nlocalverts; i++) {
		int old = get_old_label(i);
		PRINTLN_RANK("undo relabel new %d -> old %d", i, old)
		pred_temp[old] = get_old_label(pred[i]);
	}
	memcpy(pred, pred_temp, g_ptr->nlocalverts * sizeof(int64_t));
}

