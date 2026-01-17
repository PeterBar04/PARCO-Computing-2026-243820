#ifndef MATRIXMANAGER_H
#define MATRIXMANAGER_H

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <string>

using namespace std;

//Matrix in COO format read from .mtx
class MatrixCOO{
    public:
	int rows;
	int cols;
	int nnz;
	
	//COO format
	vector<int> row_index; 		//row index array
	vector<int> column_index;   //column index array
	vector<double> data;	
	
	void print();
    void readMatrixFromFile(const string& filename);

};

//---------------------------------------//
//Matrix in CSR format
class MatrixCSR{
    private:
	vector<int> pointer; //row  array
	vector<int> index;   //column index array
	vector<double> data;
	
    public:
	void print();
    void convertCOOInCSR(MatrixCOO& coo);
};


#endif
