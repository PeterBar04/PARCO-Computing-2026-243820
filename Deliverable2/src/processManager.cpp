    #include "processManager.h"

// Metadata for Scatterv (Counts and Displacements)
struct CommMetadata {
    vector<int> nnz_counts;
    vector<int> nnz_displs;
    vector<int> len_counts;
    vector<int> len_displs;

    // Helper to resize all vectors at once
    void init(int worldSize) {
        nnz_counts.assign(worldSize, 0);
        nnz_displs.assign(worldSize, 0);
        len_counts.assign(worldSize, 0);
        len_displs.assign(worldSize, 0);
    }
};

// Buffers for Shuffling (Only Root will fill these)
struct ShuffledBuffers {
    vector<int> lengths;
    vector<int> indices;
    vector<double> data;

    void allocate(int total_rows, int total_nnz) {
        lengths.assign(total_rows, 0);
        indices.assign(total_nnz, 0);
        data.assign(total_nnz, 0.0);
    }
};

// ----------------------------------------------------------------- //
// Constructor and destructor

Process::Process(int argc, char** argv) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

        localCSR.nrows = 0;
    }

Process::~Process() {
    MPI_Finalize();
}

// ----------------------------------------------------------------- //
// Methods implementation

void Process::setupData(string matrixName){
    if(rank==ROOT_RANK){
        
        globalCoo.readMatrixFromFile(matrixName);
        globalCSR.convertCOOInCSR(globalCoo);
        
        total_rows = globalCoo.rows; 
        total_nnz = globalCoo.nnz;

        globalX.resize(globalCoo.cols);

        //Initialize vector for multiplication with random values from 1 to 100
        initGlobalX(globalX, globalCoo.cols);

        globalCSR.print();
        cout << "Multiply vector: ";
        for(auto i: globalX){
            cout << i << " ";
        } cout << endl;

        //printf("[MPI process %d] I am the broadcast root, and send value %d.\n", proc.rank, total_rows);
    }

    MPI_Bcast(&total_rows, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD); //broadcast total_rows
    MPI_Bcast(&total_nnz, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD); //broadcast total_nnz

    /*
    if(proc.rank!=ROOT_RANK){
        printf("[MPI process %d] I am a broadcast receiver, and obtained value %d.\n", proc.rank, total_rows);
    }
    */
}

void Process::scatterMatrix() {

    // Every rank needs metadata to participate in MPI_Scatterv
    CommMetadata meta;
    ShuffledBuffers shuffled;
    meta.init(worldSize);

    // buffer that stores lenght of rows
    vector<int> local_lengths;

    calculateLocalRows();
    calculateCountDispl(meta);
    
    if (rank == ROOT_RANK) {
        shuffled.allocate(total_rows, total_nnz);
        //Rank 0 organizes the full_idx and full_val so they are grouped by rank.
        shuffleVectors(meta, shuffled);
    }
    
    scattering(meta, shuffled, local_lengths);

    buildCSRpointer(local_lengths);

    

    //scatterVectorX();
}


void Process::calculateLocalRows(){
    //Every rank calculates its number of local rows
    for (int i = 0; i < total_rows; i++) {
        if (i % worldSize == rank) localCSR.nrows++;
    }
}

void Process::calculateCountDispl(CommMetadata &meta){
    // Calculate row length counts/displ (len_counts) for ALL ranks
    for(int i=0; i<worldSize; i++) {
        meta.len_counts[i] = (total_rows / worldSize) + (i < (total_rows % worldSize) ? 1 : 0);
        if(i > 0) meta.len_displs[i] = meta.len_displs[i-1] + meta.len_counts[i-1];
    }

    // Calculate NNZ counts/displ (ONLY ROOT needs to loop through the matrix)
    if(rank == ROOT_RANK){
        for (int i = 0; i < total_rows; i++) {
            int target = i % worldSize;
            int nnzInRow = globalCSR.pointer[i+1] - globalCSR.pointer[i];
            meta.nnz_counts[target] += nnzInRow;
        }
        
        // Calculate displacements (offsets)
        for (int i = 1; i < worldSize; i++) {
            meta.nnz_displs[i] = meta.nnz_displs[i-1] + meta.nnz_counts[i-1];
        }
    }
    
}

void Process::shuffleVectors(CommMetadata &meta, ShuffledBuffers &shuffled){

    // Track where we are writing for each rank within the shuffled buffers
    vector<int> current_pos = meta.nnz_displs; 
    vector<int> write_ptr = meta.len_displs;

    for (int i = 0; i < total_rows; i++) {
        int target = i % worldSize;
        int row_start = globalCSR.pointer[i];
        int nnzInRow = globalCSR.pointer[i+1] - globalCSR.pointer[i];

        // Copy indices and values into the rank's assigned block
        copy(globalCSR.index.begin() + row_start, 
            globalCSR.index.begin() + row_start + nnzInRow, 
            shuffled.indices.begin() + current_pos[target]);
        
        copy(globalCSR.data.begin() + row_start, 
            globalCSR.data.begin() + row_start + nnzInRow, 
            shuffled.data.begin() + current_pos[target]);

        current_pos[target] += nnzInRow;

        shuffled.lengths[write_ptr[target]] = nnzInRow;
        write_ptr[target]++;
    }
}

void Process::scattering(CommMetadata &meta, ShuffledBuffers &shuffled, vector<int> &local_lengths) {

    // Every rank needs to know its own NNZ count to resize its local arrays
    MPI_Scatter(meta.nnz_counts.data(), 1, MPI_INT, &localCSR.nnz, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

    localCSR.index.resize(localCSR.nnz);
    localCSR.data.resize(localCSR.nnz);
    local_lengths.resize(localCSR.nrows);

    MPI_Scatterv(shuffled.indices.data(), meta.nnz_counts.data(), meta.nnz_displs.data(), MPI_INT,
                localCSR.index.data(), localCSR.nnz, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

    MPI_Scatterv(shuffled.data.data(), meta.nnz_counts.data(), meta.nnz_displs.data(), MPI_DOUBLE,
             localCSR.data.data(), localCSR.nnz, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD);

    MPI_Scatterv(shuffled.lengths.data(), meta.len_counts.data(), meta.len_displs.data(), MPI_INT,
             local_lengths.data(), localCSR.nrows, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

}

// Rebuild the local CSR pointer array
void Process::buildCSRpointer(vector<int> &local_lengths){
    localCSR.pointer.assign(localCSR.nrows + 1, 0);
    for (int i = 0; i < localCSR.nrows; i++) {
        localCSR.pointer[i+1] = localCSR.pointer[i] + local_lengths[i];
    }
}


void Process::initGlobalX(vector<double>& v, int cols){
    random_device rd;  //Obtain a random number from hardware to seed the generator
    mt19937 gen(rd()); //Initialize the Mersenne Twister engine with the seed
    uniform_real_distribution<double> dis(std::nextafter(0.0, 100.0), 100.0); //Define the range (nextafter ensures 0 is not included)

    for (int i = 0; i < cols; i++) {
        v[i] = dis(gen);
    }
}


void Process::distributeVectorX() {
    // 1. Calculate how many x elements each rank gets (Metadata)
    vector<int> x_counts(worldSize, 0);
    vector<int> x_displs(worldSize, 0);

    // Cyclic calculation: Who owns which column index?
    // Note: Use total_cols (which should be equal to total_rows for square matrices)
    int total_cols = total_rows; 

    for (int j = 0; j < total_cols; j++) {
        x_counts[j % worldSize]++;
    }

    // Calculate displacements for the scatter
    for (int i = 1; i < worldSize; i++) {
        x_displs[i] = x_displs[i-1] + x_counts[i-1];
    }

    // 2. Rank 0 Shuffles the global vector into contiguous blocks
    vector<double> shuffled_x;
    if (rank == ROOT_RANK) {
        shuffled_x.resize(total_cols);
        
        // Use a temporary copy of displacements as "writing heads"
        vector<int> current_pos = x_displs; 
        
        for (int j = 0; j < total_cols; j++) {
            int target = j % worldSize;
            shuffled_x[current_pos[target]] = globalX[j];
            current_pos[target]++;
        }
    }

    // 3. Prepare local storage
    int my_x_count = x_counts[rank];
    localX.resize(my_x_count);

    // 4. Scatter the shuffled vector to everyone
    MPI_Scatterv(shuffled_x.data(), x_counts.data(), x_displs.data(), MPI_DOUBLE,
                 localX.data(), my_x_count, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD);
}


void Process::scatterVectorX(){

    // 1. Setup containers
    vector<vector<int>> requests;
    map<int, int> ghost_map;
    vector<double> ghost_buffer;
    vector<int> send_counts, recv_counts;
    vector<double> y_local;

    // 2. Run the phases
    cout << endl << "----- IDENTIFY PHASE -------" << endl;
    identifyGhostEntries(requests, ghost_map);

    cout << endl << "----- EXCHANGE METADATA -------" << endl;
    exchangeMetadata(requests, send_counts, recv_counts);

    exchangeGhostValues(requests, send_counts, recv_counts, ghost_buffer);

    cout << endl << "----- EXCHANGE VALUES -------" << endl;
    // --- DEBUG PRINT: Ghost Values (The Result) ---
for (int r = 0; r < worldSize; r++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == r) {
        printf("\n=== Rank %d Ghost Exchange Results ===\n", rank);
        
        // 1. Print raw buffer contents
        printf("Raw Ghost Buffer (Size %lu):\n  [ ", ghost_buffer.size());
        for (double val : ghost_buffer) {
            printf("%.2f ", val);
        }
        printf("]\n");

        // 2. Print interpreted values (Global Index -> Value)
        printf("Decoded Values (Using Ghost Map):\n");
        if (ghost_map.empty()) {
            printf("  (No ghost entries needed)\n");
        } else {
            // ghost_map stores: Global_Index -> Position_in_Buffer
            for (auto const& [global_idx, buffer_pos] : ghost_map) {
                // Safety check
                if (buffer_pos < ghost_buffer.size()) {
                    double val = ghost_buffer[buffer_pos];
                    printf("  Global Index %d = %.2f (stored at buffer[%d])\n", 
                           global_idx, val, buffer_pos);
                } else {
                    printf("  ERROR: Global Index %d maps to invalid buffer position %d\n", 
                           global_idx, buffer_pos);
                }
            }
        }
        fflush(stdout);
    }
}
MPI_Barrier(MPI_COMM_WORLD);
// ----------------------------------------------
    cout << endl << "----- SPMV -------" << endl;
    performLocalSpMV(ghost_map, ghost_buffer, y_local);
}


// =========================================================================
// PHASE 1: Identify what data we are missing ("Ghost Entries")
// =========================================================================
void Process::identifyGhostEntries(vector<vector<int>>& requests, map<int, int>& ghost_map) {
    // requests[target_rank] will store the list of global indices we need from them
    requests.resize(worldSize);
    
    // Use a set to avoid asking for the same index twice
    set<int> unique_ghosts;

    for (int col : localCSR.index) {
        // If I don't own this column, I need to ask for it
        if (col % worldSize != rank) {
            unique_ghosts.insert(col);
        }
    }

    // Sort them to ensure deterministic order (good for debugging)
    // and map them to a specific slot in our future 'ghost_buffer'
    int buffer_pos = 0;
    for (int global_col : unique_ghosts) {
        int owner = global_col % worldSize; //which rank has this colum?
        requests[owner].push_back(global_col); //at rank n will be requested the col
        
        // Save where this specific global index will live in our receive buffer
        ghost_map[global_col] = buffer_pos; 
        buffer_pos++;
    }

    // --- DEBUG PRINT: Requests and Ghost Map ---
    for (int r = 0; r < worldSize; r++) {
    MPI_Barrier(MPI_COMM_WORLD); // Wait for turn
    if (rank == r) {
        printf("\n=== Rank %d Identification Phase ===\n", rank);
        
        // 1. Print Requests (Who am I asking and for what?)
        printf("Requests to other ranks:\n");
        bool asking_anyone = false;
        for (int target = 0; target < worldSize; target++) {
            if (!requests[target].empty()) {
                asking_anyone = true;
                printf("  -> To Rank %d: [ ", target);
                // Print up to 10 indices to keep it readable
                int limit = (requests[target].size() > 10) ? 10 : requests[target].size();
                for (int i = 0; i < limit; i++) {
                    printf("%d ", requests[target][i]);
                }
                if (requests[target].size() > 10) printf("... (%lu total)", requests[target].size());
                printf("]\n");
            }
        }
        if (!asking_anyone) printf("  (None - I have all data locally)\n");

        // 2. Print Ghost Map (How do I find them later?)
        printf("Ghost Map (Global Index -> Buffer Position):\n");
        if (ghost_map.empty()) {
            printf("  (Empty)\n");
        } else {
            int count = 0;
            printf("  { ");
            for (auto const& [global_idx, buffer_pos] : ghost_map) {
                printf("%d:%d ", global_idx, buffer_pos);
                if (++count >= 10) { 
                    printf("... "); 
                    break; 
                }
            }
            printf("}\n");
        }
        fflush(stdout);
    }
    }
    MPI_Barrier(MPI_COMM_WORLD); // Final sync
// -------------------------------------------
}


// =========================================================================
// PHASE 2: Exchange Metadata (Tell neighbors how much we need)
// =========================================================================
void Process::exchangeMetadata(const vector<vector<int>>& requests, 
                               vector<int>& num_indices_I_request, 
                               vector<int>& num_indices_others_need_from_me) {
    
    num_indices_I_request.resize(worldSize);
    num_indices_others_need_from_me.resize(worldSize);

    // Fill the send buffer: "I am going to ask Rank i for X indices"
    for (int i = 0; i < worldSize; i++) {
        num_indices_I_request[i] = requests[i].size();
    }

    // Exchange counts. 
    // After this, num_indices_others_need_from_me[i] tells us:
    // "Rank i is about to send me a list of Y indices it wants."
    MPI_Alltoall(num_indices_I_request.data(), 1, MPI_INT, 
                 num_indices_others_need_from_me.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // --- DEBUG PRINT: Metadata Exchange (The Handshake) ---
    for (int r = 0; r < worldSize; r++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == r) {
        printf("\n=== Rank %d Metadata Phase ===\n", rank);
        
        // 1. What I am asking for (My Requests)
        printf("I am asking others for this many indices:\n");
        printf("  [ ");
        for (int count : num_indices_I_request) printf("%d ", count);
        printf("]\n");

        // 2. What others are asking me for (My Obligations)
        printf("Others are asking me for this many indices:\n");
        printf("  [ ");
        for (int count : num_indices_others_need_from_me) printf("%d ", count);
        printf("]\n");

        fflush(stdout);
    }
}
MPI_Barrier(MPI_COMM_WORLD);
// -----------------------------------------------------

}


// =========================================================================
// PHASE 3: The Handshake (Exchange Indices, then Exchange Values)
// =========================================================================
void Process::exchangeGhostValues(const vector<vector<int>>& requests, 
                                  const vector<int>& num_indices_I_request, 
                                  const vector<int>& num_indices_others_need_from_me,
                                  vector<double>& ghost_buffer) {

    // --- STEP A: PREPARE DISPLACEMENTS FOR MPI ---
    vector<int> send_displs(worldSize, 0); // For outgoing requests
    vector<int> recv_displs(worldSize, 0); // For incoming requests
    
    int total_requests_outgoing = 0;
    int total_requests_incoming = 0;

    for (int i = 0; i < worldSize; i++) {
        send_displs[i] = (i == 0) ? 0 : send_displs[i-1] + num_indices_I_request[i-1];
        recv_displs[i] = (i == 0) ? 0 : recv_displs[i-1] + num_indices_others_need_from_me[i-1];
        
        total_requests_outgoing += num_indices_I_request[i];
        total_requests_incoming += num_indices_others_need_from_me[i];
    }

    // Flatten my 2D requests into a 1D array for MPI
    vector<int> flat_requests_outgoing;
    flat_requests_outgoing.reserve(total_requests_outgoing);
    for (const auto& req_vec : requests) {
        flat_requests_outgoing.insert(flat_requests_outgoing.end(), req_vec.begin(), req_vec.end());
    }

    // Buffer to hold the indices OTHER people want from ME
    vector<int> indices_others_want(total_requests_incoming);


    // --- STEP B: EXCHANGE THE INDICES (Ask for what we want) ---
    // We send 'flat_requests_outgoing' -> Others receive into 'indices_others_want'
    MPI_Alltoallv(flat_requests_outgoing.data(), num_indices_I_request.data(), send_displs.data(), MPI_INT,
                  indices_others_want.data(), num_indices_others_need_from_me.data(), recv_displs.data(), MPI_INT, 
                  MPI_COMM_WORLD);


    // --- STEP C: FETCH THE VALUES FROM MY LOCAL STORAGE ---
    // Now I know exactly which Global Indices others want. I must look them up.
    vector<double> values_others_need(total_requests_incoming);

    for (int i = 0; i < total_requests_incoming; i++) {
        int global_idx_requested = indices_others_want[i];
        
        // Convert Global Index -> Local Index (Cyclic Partitioning Logic)
        // If I own Global 4, 8, 12... and worldSize is 4:
        // Global 4 is at local index 1 (4 / 4 = 1).
        int local_idx = global_idx_requested / worldSize; 

        // Safety check (optional but recommended)
        if (local_idx < 0 || local_idx >= localX.size()) {
            printf("Error on Rank %d: Requested global %d maps to invalid local %d\n", rank, global_idx_requested, local_idx);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        values_others_need[i] = localX[local_idx];
    }

    // --- STEP D: SEND THE VALUES BACK (Reply to requests) ---
    ghost_buffer.resize(total_requests_outgoing);

    // Note carefully: The send/recv counts are FLIPPED here compared to Step B.
    // We are sending data back to satisfy the requests we just received.
    MPI_Alltoallv(values_others_need.data(), num_indices_others_need_from_me.data(), recv_displs.data(), MPI_DOUBLE,
                  ghost_buffer.data(), num_indices_I_request.data(), send_displs.data(), MPI_DOUBLE, 
                  MPI_COMM_WORLD);
}

// =========================================================================
// PHASE 4: Perform Local SpMV (Compute y = Ax)
// =========================================================================
void Process::performLocalSpMV(const map<int, int>& ghost_map, 
                               const vector<double>& ghost_buffer, 
                               vector<double>& y_local) {
    
    y_local.assign(localCSR.nrows, 0.0);

    for (int i = 0; i < localCSR.nrows; i++) {
        double sum = 0.0;
        
        for (int k = localCSR.pointer[i]; k < localCSR.pointer[i+1]; k++) {
            int global_col = localCSR.index[k];
            double x_val;

            // CHECK: Is this column local or ghost?
            if (global_col % worldSize == rank) {
                // I own it! Math: local_index = global / worldSize
                x_val = localX[global_col / worldSize];
            } else {
                // It's a ghost! Use the map to find where we put it in the buffer
                // .at() is safer than [] because it throws an error if missing
                x_val = ghost_buffer[ghost_map.at(global_col)];
            }
            
            sum += localCSR.data[k] * x_val;
        }
        y_local[i] = sum;
    }

    // --- DEBUG PRINT: Local Result Y ---
    for (int r = 0; r < worldSize; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == r) {
            printf("\n=== Rank %d Local Y (Result) ===\n", rank);
            printf("Computed Y values for my rows:\n  [ ");
            for (double val : y_local) {
                printf("%.2f ", val);
            }
            printf("]\n");
            fflush(stdout);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
// -----------------------------------
}


void Process::print(){
    // print received local_idx and local_data
    for (int r = 0; r < worldSize; r++) {
        if (rank == r) {
            printf("\n--- Rank %d | local_nrows: %d | local_nnz: %d ---\n", 
                    rank, localCSR.nrows, (int)localCSR.index.size());
            
            printf("local_pointer: [ ");
            for (int val : localCSR.pointer) printf("%d ", val);
            printf("]\n");

            printf("local_idx: [ ");
            for (int val : localCSR.index) printf("%d ", val);
            printf("]\n");

            printf("local_val: [ ");
            for (double val : localCSR.data) printf("%.2f ", val);
            printf("]\n");

            printf("local_x: [ ");
            for (double val : localX) printf("%.2f ", val);
            printf("]\n");
            
            // Push the output to the console immediately
            fflush(stdout); 
        }
        // Wait for the current rank to finish printing before the next one starts
        MPI_Barrier(MPI_COMM_WORLD);
    }
    cout << endl;
}