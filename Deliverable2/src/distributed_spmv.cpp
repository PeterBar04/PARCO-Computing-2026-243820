#include <iostream>

#include "matrixManager.h"
#include "processManager.h"

using namespace std;

//---------------------------------------//

/*
void spmv(const CSR& csr, const vector<double> &v, vector<double> &r){
    int rows = (int)r.size();

    for(int ip=0; ip<rows; ip++){		
    double sum = 0.0;
        for (int k = csr.pointer[ip]; k < csr.pointer[ip+1]; k++) {
            sum += csr.data[k] * v[ csr.index[k] ];
        }
        r[ip] = sum;
	}
}
*/



int main(int argc, char** argv) {
	
	srand(time(0));

    Process p(argc, argv);

	if (argc < 1) {
        cout << "Usage: ./distributed_spmv <matrixName>" << endl;
        return -1;
    }

    string matrixName = argv[1];

    p.setupData(matrixName);
    p.scatterMatrix();
    p.distributeVectorX();
    
    p.print();

    p.scatterVectorX();

	return 0;
}

