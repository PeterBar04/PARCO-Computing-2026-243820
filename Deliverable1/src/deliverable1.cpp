#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <cmath>
#include "timer.h"

using namespace std;

//Matrix in COO format read from .mtx
typedef struct Matrix{
	int rows;
	int cols;
	int nnz;
	
	//COO format
	vector<int> row_index; 		//row index array
	vector<int> column_index;   //column index array
	vector<double> data;	
	
	void print(){
		cout << "Matrix COO format\n" <<endl;
		cout << "row_index: ";
		for(int i: row_index){
			cout << i << " ";
		} cout << endl;
		
		cout << "column_index: ";
		for(int i: column_index){
			cout << i << " ";
		} cout << endl;
		
		cout << "data: ";
		for(int i: data){
			cout << i << " ";
		} cout << endl;
	}		
}Matrix;


//---------------------------------------//
//Matrix in CSR format
typedef struct CSR{
	vector<int> pointer; //row  array
	vector<int> index;   //column index array
	vector<double> data;
	
	void print(){
		cout << "POINTER: ";
		for(int i: pointer){
			cout << i << " ";
		} cout << endl;
		
		cout << "INDEX: ";
		for(int i: index){
			cout << i << " ";
		} cout << endl;
		
		cout << "DATA: ";
		for(int i: data){
			cout << i << " ";
		} cout << endl;
	}
};

//---------------------------------------//
//Read matrix from file and get COO format
void read_matrix_from_file(Matrix& matrix, const string& filename){
	std::ifstream file(filename);
	
 if (!file.is_open()) {
        cerr << "Error: cannot open file " << filename << endl;
        exit(1);
    }
 
	// Ignore comments headers
	while (file.peek() == '%') file.ignore(2048, '\n');
	
	// Read number of rows and columns
	file >> matrix.rows >> matrix.cols >> matrix.nnz;
	
	// fill the matrix with data
	for (int l = 0; l < matrix.nnz; l++)
	{
	    double data;
	    int row, col;
	    file >> row >> col >> data;    
	    
	    matrix.row_index.push_back(row-1); 		//mtx indexes start from 1
	    matrix.column_index.push_back(col-1);
	    matrix.data.push_back(data);
	}
	
	file.close();
}

//---------------------------------------//
//Convert Matrix from COO to CSR
void convert_COO_in_CSR(Matrix& matrix, CSR& csr){

	int nrows = matrix.rows;
    int nnz   = matrix.nnz;

    // --- Combine the vectors in a triplet
    struct Entry {
        int row;
        int col;
        double val;
    };

    vector<Entry> entries(nnz);
    for (int i = 0; i < nnz; i++) {
        entries[i] = {matrix.row_index[i], matrix.column_index[i], matrix.data[i]};
    }

    // --- Sorting the values first for rows, then for columns
    sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        if (a.row == b.row)
            return a.col < b.col;
        return a.row < b.row;
    });

    // --- Copy in the csr vector
    csr.data.resize(nnz);
    csr.index.resize(nnz);
    csr.pointer.assign(nrows + 1, 0);

    for (int i = 0; i < nnz; i++) {
        csr.data[i] = entries[i].val;
        csr.index[i] = entries[i].col;
        csr.pointer[entries[i].row + 1]++; 
    }

    for (int i = 0; i < nrows; i++) {
        csr.pointer[i + 1] += csr.pointer[i];
    }

}

//---------------------------------------//
//Initialize vector for multiplication with random values from 1 to 100
void init_vec(vector<double> &v, int cols){
	for(int i=0; i<cols; i++){
		v[i] = rand() % 100 + 1;
	}
}

//---------------------------------------//
//Multiply matrix and vector
#ifdef _OPENMP
double parallel_matrix_vector_mul(const CSR& csr, const vector<double> &v, vector<double> &r,
						const string &schedule_type, int chunk_size){

	int rows = (int)r.size();
	
  omp_sched_t sched_kind;
  if (schedule_type == "static") sched_kind = omp_sched_static;
  else if (schedule_type == "dynamic") sched_kind = omp_sched_dynamic;
  else if (schedule_type == "guided") sched_kind = omp_sched_guided;
  else sched_kind = omp_sched_auto;

  omp_set_schedule(sched_kind, chunk_size);
	
	double start, end;
	GET_TIME(start);
	
	#pragma omp parallel for schedule(runtime)
	for(int ip=0; ip<rows; ip++){		
		double sum = 0.0;
		    for (int k = csr.pointer[ip]; k < csr.pointer[ip+1]; k++) {
		        sum += csr.data[k] * v[ csr.index[k] ];
		    }
		    r[ip] = sum;
	}
	
	GET_TIME(end);
	
	return (end-start)*1000;
}
#endif

double serial_matrix_vector_mul(const CSR& csr, const vector<double> &v, vector<double> &r){

	int rows = (int)r.size();
	
	double start, end;
	GET_TIME(start);
	
	for(int ip=0; ip<rows; ip++){		
		double sum = 0.0;
		    for (int k = csr.pointer[ip]; k < csr.pointer[ip+1]; k++) {
		        sum += csr.data[k] * v[ csr.index[k] ];
		    }
		    r[ip] = sum;
	}
	
	GET_TIME(end);
	
	return (end-start)*1000;	
	
}



int main(int argc, char** argv) {
	
	srand(time(0));
	
	if (argc < 4) {
        cout << "Usage: ./deliverable1 <matrix_file> <schedule_type> <chunk_size> <num_runs>" << endl;
        return -1;
    }
	
    string matrix_filename = argv[1];
	string schedule_type = argv[2];
    int chunk_size = atoi(argv[3]);
    int num_runs = atoi(argv[4]);	
 
	Matrix matrix;
	CSR csr;

	read_matrix_from_file(matrix, matrix_filename);
	
	vector<double> vec(matrix.cols, 0); 
	vector<double> result(matrix.cols, 0); 
	
	convert_COO_in_CSR(matrix,csr);
	
	init_vec(vec,matrix.cols);

  int i;
  double t;
  vector<double> times;
  
  for(i=0;i<num_runs;i++){

    #ifdef _OPENMP
    	t = parallel_matrix_vector_mul(csr, vec, result, schedule_type, chunk_size);
      times.push_back(t);
  	#else
      t = serial_matrix_vector_mul(csr, vec, result);
      times.push_back(t);
    #endif
   }
   
 //calculate the p90
  sort(times.begin(), times.end());
  double pos = 0.9 * (times.size() - 1);
  int lower = (int)floor(pos);
  int upper = (int)ceil(pos);
  
  double p90;
  if (lower == upper)
      p90 = times[lower];
  else
    p90 = times[lower] + (times[upper] - times[lower]) * (pos - lower);
  
  
  printf("P90 %e\n", p90);
  
	return 0;
}
