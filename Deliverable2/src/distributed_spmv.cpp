#include <iostream>
#include <mpi.h>

#include "matrixManager.h"
#include "processManager.h"

using namespace std;


int main(int argc, char** argv) {

    Process p(argc, argv);

	if (argc < 1) {
        cout << "Usage: ./distributed_spmv <matrixName>" << endl;
        return -1;
    }

    string matrixName = argv[1];

    p.setupData(matrixName);
    
    p.scatterData();
    
    p.exchangeVectorX();

    // 1. Start Timer
    MPI_Barrier(MPI_COMM_WORLD); // Sync before starting
    double execution_start_time = MPI_Wtime();

    // 2. Run the math
    p.performLocalSpMV();

    // 3. Stop Timer
    MPI_Barrier(MPI_COMM_WORLD); // Sync to ensure slowest rank finishes
    double execution_end_time = MPI_Wtime();

    // 4. Print ONLY from Rank 0 with a unique "Tag"
    if (p.getRank() == ROOT_RANK) {
        double execution_time = execution_end_time - execution_start_time;
        // "EXEC_TIME" is the keyword we will look for in Bash
        printf("EXEC_TIME %.6f COMM_TIME %.6f\n", execution_time, p.getCommTime());
}

    //p.print();

	return 0;
}

