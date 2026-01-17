#include <iostream>
#include <mpi.h>
#include <time.h>

#include "matrixManager.h"

using namespace std;


int main(int argc, char** argv) {
	
	srand(time(0));
	
	if (argc < 1) {
        cout << "Usage: ./distributed_spmv <matrixName>" << endl;
        return -1;
    }
	
    string matrixName = argv[1];
 
	MatrixCOO coo;
	MatrixCSR csr;

    coo.readMatrixFromFile(matrixName);
    csr.convertCOOInCSR(coo);
	
	vector<double> vec(coo.cols, 0); 
	vector<double> result(coo.cols, 0); 
	
    //coo.print();
    //csr.print();
  
	return 0;
}

