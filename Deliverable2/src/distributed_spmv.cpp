#include <iostream>

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

    p.performLocalSpMV();

    p.print();

	return 0;
}

