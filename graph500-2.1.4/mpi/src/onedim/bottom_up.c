#include "bottom_up.h"

#include <string.h>

#include "oned_csr.h"

#include "constants.h"
#include "bfs.h"
#include "print.h"

extern oned_csr_graph g;
extern int64_t *pred;

extern int64_t *frontier;
extern int64_t *frontier_next;

void one_step_bottom_up() {
    int i;
    for (i = 0; i < g.nlocalverts; i++) {
#ifdef SHOWDEBUG
        PRINTLN_RANK("checking vertex %d ...", VERTEX_TO_GLOBAL(rank, i))
#endif
        if (pred[i] == -1) {
#ifdef SHOWDEBUG
            PRINTLN_RANK("checking vertex %d yes", VERTEX_TO_GLOBAL(rank, i))
#endif
            int j;
            for (j = (int) g.rowstarts[i]; j < g.rowstarts[i + 1]; j++) {
                int64_t parent_global = g.column[j];
#ifdef SHOWDEBUG
                PRINTLN_RANK("checking vertex %d - %d ...", VERTEX_TO_GLOBAL(rank, i), parent_global)
#endif
                if (TEST_GLOBAL(parent_global, frontier)) {
#ifdef SHOWDEBUG
                    PRINTLN_RANK("checking vertex %d - %d yes", VERTEX_TO_GLOBAL(rank, i), parent_global)
#endif
                    pred[i] = parent_global;
                    SET_GLOBAL(VERTEX_TO_GLOBAL(rank, i), frontier_next);
                    break;
                }
#ifdef SHOWDEBUG
                else
                PRINTLN_RANK("checking vertex %d - %d no.", VERTEX_TO_GLOBAL(rank, i), parent_global)
#endif
            }
        }
    }
}
