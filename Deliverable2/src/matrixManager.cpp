#include "matrixManager.h"

#include <algorithm> // Required for std::min

void MatrixCOO::print() {
    // 1. Determine the limit (10 or the actual size, whichever is smaller)
    // We cast .size() to int to avoid warnings
    int limit = std::min((int)row_index.size(), 10); 

    cout << "Matrix COO format (Preview first " << limit << " entries)\n" << endl;

    // --- ROWS ---
    cout << "rows: ";
    for(int i = 0; i < limit; i++){
        cout << row_index[i] << " ";
    }
    // Visual indicator if there are more elements
    if (row_index.size() > 10) cout << "... (" << row_index.size() << " total)";
    cout << endl;

    // --- COLS ---
    cout << "cols: ";
    for(int i = 0; i < limit; i++){
        cout << column_index[i] << " ";
    }
    if (column_index.size() > 10) cout << "...";
    cout << endl;

    // --- DATA ---
    cout << "data: ";
    for(int i = 0; i < limit; i++){
        cout << data[i] << " ";
    }
    if (data.size() > 10) cout << "...";
    cout << endl;
    
    cout << endl;
}

#include <algorithm> // Required for std::min
#include <iostream>

void MatrixCSR::print() {
    // 1. Calculate limits
    // pointer vector size is usually (rows + 1)
    int ptr_limit = std::min((int)pointer.size(), 10);
    
    // index and data vectors size is NNZ (Total Non-Zeros)
    int nnz_limit = std::min((int)index.size(), 10);

    cout << "Matrix CSR format (Preview first 10 entries)\n" << endl;

    // --- POINTER ARRAY ---
    cout << "POINTER: ";
    for(int i = 0; i < ptr_limit; i++){
        cout << pointer[i] << " ";
    }
    if (pointer.size() > 10) cout << "... (" << pointer.size() << " total)";
    cout << endl;
    
    // --- INDEX ARRAY ---
    cout << "INDEX  : ";
    for(int i = 0; i < nnz_limit; i++){
        cout << index[i] << " ";
    }
    if (index.size() > 10) cout << "... (" << index.size() << " total)";
    cout << endl;
    
    // --- DATA ARRAY ---
    cout << "DATA   : ";
    for(int i = 0; i < nnz_limit; i++){
        cout << data[i] << " ";
    }
    if (data.size() > 10) cout << "... (" << data.size() << " total)";
    cout << endl;
    
    cout << endl;
}

//Read matrix from file and get COO format
void MatrixCOO::readMatrixFromFile(const string& filename){
	std::ifstream file(filename);
	
 if (!file.is_open()) {
        cerr << "Error: cannot open file " << filename << endl;
        exit(1);
    }
 
	// Ignore comments headers
	while (file.peek() == '%') file.ignore(2048, '\n');
	
	// Read number of rows and columns
	file >> rows >> cols >> nnz;
	
	// fill the matrix with data
	for (int l = 0; l < nnz; l++)
	{
	    double dataFile;
	    int row, col;
	    file >> row >> col >> dataFile;    
	    
	    row_index.push_back(row-1); 		//mtx indexes start from 1
	    column_index.push_back(col-1);
	    data.push_back(dataFile);
	}
	
	file.close();
}

//---------------------------------------//
//Convert Matrix from COO to CSR
void MatrixCSR::convertCOOInCSR(MatrixCOO& coo){

	nrows = coo.rows;
    nnz   = coo.nnz;

    // --- Combine the vectors in a triplet
    struct Entry {
        int row;
        int col;
        double val;
    };

    vector<Entry> entries(nnz);
    for (int i = 0; i < nnz; i++) {
        entries[i] = {coo.row_index[i], coo.column_index[i], coo.data[i]};
    }

    // --- Sorting the values first for rows, then for columns
    sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        if (a.row == b.row)
            return a.col < b.col;
        return a.row < b.row;
    });

    // --- Copy in the csr vector
    data.resize(nnz);
    index.resize(nnz);
    pointer.assign(nrows + 1, 0);

    for (int i = 0; i < nnz; i++) {
        data[i] = entries[i].val;
        index[i] = entries[i].col;
        pointer[entries[i].row + 1]++; 
    }

    for (int i = 0; i < nrows; i++) {
        pointer[i + 1] += pointer[i];
    }

}