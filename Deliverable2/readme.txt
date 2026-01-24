. Why use Cyclic Partitioning?

Professors often suggest this because it is excellent for Load Balancing. If the "top" of your matrix is very dense and the "bottom" is very sparse, Block partitioning would leave some processes working harder than others. Cyclic partitioning "samples" rows from the whole matrix, making it likely that every process gets a similar total number of non-zero entries.

We observe a significant latency jump between NP=1 and NP=2. This is expected because NP=1 runs entirely in local memory with contiguous access. At NP=2, the cyclic partition of a random matrix forces 50% of accesses to use the ghost_map (a red-black tree), which introduces $O(\log N)$ lookup overhead compared to the $O(1)$ array access at NP=1.
At NP=1: Your code is essentially a serial program. The CPU predicts the branch if (col % size == rank) correctly 100% of the time, and you never do the slow map lookup.
