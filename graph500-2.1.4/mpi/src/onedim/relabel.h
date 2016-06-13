
#ifndef RELABEL_H
#define RELABEL_H

#include "oned_csr.h"

int64_t get_new_label(int64_t old);

int64_t get_old_label(int64_t new_lbl);

void relabel(oned_csr_graph *g);

void undo_relabel(int64_t* pred);	

#endif // RELABEL_H