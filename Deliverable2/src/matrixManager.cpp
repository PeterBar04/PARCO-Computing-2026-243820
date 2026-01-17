#include "matrixManager.h"

void MatrixCOO::print() {
    cout << "Matrix COO format\n" <<endl ;
    cout << "rows: ";
    for(int i: row_index){
        cout << i << " ";
    } cout << endl;
    
    cout << "cols: ";
    for(int i: column_index){
        cout << i << " ";
    } cout << endl;
    
    cout << "data: ";
    for(double i: data){
        cout << i << " ";
    } cout << endl;
    cout << endl;
}

void MatrixCSR::print(){
    cout << "Matrix CSR format\n" <<endl;
    cout << "POINTER: ";
    for(int i: pointer){
        cout << i << " ";
    } cout << endl;
    
    cout << "INDEX  : ";
    for(int i: index){
        cout << i << " ";
    } cout << endl;
    
    cout << "DATA   : ";
    for(double i: data){
        cout << i << " ";
    } cout << endl;
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

	int nrows = coo.rows;
    int nnz   = coo.nnz;

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