- Why use Cyclic Partitioning?

Professors often suggest this because it is excellent for Load Balancing. If the "top" of your matrix is very dense and the "bottom" is very sparse, Block partitioning would leave some processes working harder than others. Cyclic partitioning "samples" rows from the whole matrix, making it likely that every process gets a similar total number of non-zero entries.

- NNZ per Rank (Computation Load Balance):What it is: The number of non-zero elements stored locally on each processor. Since your loop runs once per non-zero ($y += A_{ij} \times x_j$), this represents the computational workload.Why it matters:Ideal: $Min \approx Max$. Everyone finishes at the same time.Bad: $Max \gg Avg$. One processor has way more work (the "straggler"). The entire cluster has to wait for this one slow rank to finish, wasting resources.

- Communication Volume (Network Load):What it is: The total count of double values a rank sends and receives during the Ghost Exchange.Why it matters:Unstructured (Random) Matrices: You will see High Volume. Ranks need data from everywhere.Structured (Banded) Matrices: You will see Low Volume. Ranks only need data from immediate neighbors (Rank $i-1$, $i+1$).

- We observe a significant latency jump between NP=1 and NP=2. This is expected because NP=1 runs entirely in local memory with contiguous access. At NP=2, the cyclic partition of a random matrix forces 50% of accesses to use the ghost_map (a red-black tree), which introduces $O(\log N)$ lookup overhead compared to the $O(1)$ array access at NP=1.
At NP=1: Your code is essentially a serial program. The CPU predicts the branch if (col % size == rank) correctly 100% of the time, and you never do the slow map lookup.
 
hanging from MAP to UNORDERED MAP, I increased the performance.
Old (std::map): NP=1 (4.4ms), NP=2 (31.7ms).Slowdown: ~7.2x. The code broke when moving to distributed memory.
New (std::unordered_map): NP=1 (3.45ms), NP=2 (4.95ms).Slowdown: ~1.4x. This is completely normal overhead for checking "Is this local or ghost?"Conclusion: The $O(1)$ hash table lookup fixed the CPU bottleneck.
