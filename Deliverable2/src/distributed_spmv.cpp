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
    
    p.exchangeGhostIdentifier();

    p.prepareOptimizedStructures();

    p.runCalculation(100); //100 = num of iteration

    p.printMetrics();

    // 4. Print ONLY from Rank 0 with a unique "Tag"
    if (p.getRank() == ROOT_RANK) {
        // "EXEC_TIME" is the keyword we will look for in Bash
        printf("EXEC_TIME %.6f COMM_TIME %.6f\n", p.getCompTime(), p.getCommTime());
    }

	return 0;
}

