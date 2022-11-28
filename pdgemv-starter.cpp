#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <mpi.h>
#include <cassert>

using namespace std;

const bool DEBUG = false;

// initialize matrix and vectors (A is mxn, x is xn-vec)
void init_rand(double* a, int m, int n, double* x, int xn);
// local matvec: y = y+A*x, where A is m x n
void local_gemv(double* A, double* x, double* y, int m, int n);

int main(int argc, char** argv) {

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    int nProcs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(rank*12345);

    // Read dimensions and processor grid from command line arguments
    if(argc != 6) {
        cerr << "Usage: ./a.out rows cols pr pc identity[yes|no])" << endl;
        return 1;
    }

    int m, n, pr, pc;
    m  = atoi(argv[1]);
    n  = atoi(argv[2]);
    pr = atoi(argv[3]);
    pc = atoi(argv[4]);
	bool IDENT = !strcmp(argv[5], "yes");

	if (IDENT && m != n) {
		cerr << "If using an identity matrix, number of rows and columns must be equal" << endl;
		return 1;
	}

    if(pr*pc != nProcs) {
        cerr << "Processor grid doesn't match number of processors" << endl;
        return 1;
    }
    if(m % pr || n % pc || m % nProcs || n % nProcs) {
        cerr << "Processor grid doesn't divide rows and columns evenly" << endl;
        return 1;
    }

    // Set up row and column communicators
    int ranki = rank % pr; // proc row coordinate
    int rankj = rank / pr; // proc col coordinate
    
    // Create row and column communicators using MPI_Comm_split
    MPI_Comm row_comm, col_comm;
   	MPI_Comm_split(MPI_COMM_WORLD, rankj, rank, &col_comm);
	MPI_Comm_split(MPI_COMM_WORLD, ranki, rank, &row_comm);

    // Check row and column communicators and proc coordinates
    int rankichk, rankjchk;
    MPI_Comm_rank(row_comm,&rankjchk);
    MPI_Comm_rank(col_comm,&rankichk);

    if(ranki != rankichk || rankj != rankjchk) {
        cerr << "Processor ranks are not as expected, check row and column communicators" << endl;
        return 1;
    }

    // Initialize matrices and vectors
    int mloc = m / pr;     // number of rows of local matrix
    int nloc = n / pc;     // number of cols of local matrix
    int ydim = m / nProcs; // number of entries of local output vector
    int xdim = n / nProcs; // number of entries of local input vector
    double* Alocal = new double[mloc*nloc];
    double* xlocal = new double[xdim];
    double* ylocal = new double[ydim];

   	init_rand(Alocal, mloc, nloc, xlocal, xdim);
	if (IDENT) {
		for (int i = 0; i < mloc; i++) {
			for (int j = 0; j < nloc; j++) {
				if (i + ranki*mloc == j + rankj*nloc) {
					Alocal[i+j*mloc] = 1;
				} else {
					Alocal[i+j*mloc] = 0;
				}
			}
		}
	}
    memset(ylocal,0,ydim*sizeof(double));

    // start timer
    double gather_time, reduce_time, comp_time, redist_time, total_time, start = MPI_Wtime();

	double* recv_buffer = new double[nloc];
    // Communicate input vector entries
	MPI_Allgather(xlocal, xdim, MPI_DOUBLE, recv_buffer, xdim, MPI_DOUBLE, col_comm);
	gather_time = MPI_Wtime() - start;

    // Perform local matvec
	double* matvecmul_result = new double[mloc];
	memset(matvecmul_result, 0, mloc*sizeof(double));
	local_gemv(Alocal, recv_buffer, matvecmul_result, mloc, nloc);
   	MPI_Barrier(MPI_COMM_WORLD);
	comp_time = MPI_Wtime() - gather_time - start;

	// Communicate output vector entries
	int* revcounts = new int[pc];
	for (int i = 0; i < pc; i++) 
		revcounts[i] = ydim;
	MPI_Reduce_scatter(matvecmul_result, ylocal, revcounts, MPI_DOUBLE, MPI_SUM, row_comm);
   	reduce_time = MPI_Wtime() - gather_time - comp_time - start;

	// Bonus: redistribute the output vector to match input vector
   	int dest_rank = ranki * pc + rankj;
	MPI_Request request;
	MPI_Isend(ylocal, ydim, MPI_DOUBLE, dest_rank, 0, MPI_COMM_WORLD, &request);
	MPI_Status status;
	MPI_Recv(ylocal, ydim, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	
	// Stop timer
    MPI_Barrier(MPI_COMM_WORLD);
   	redist_time = MPI_Wtime() - reduce_time - comp_time - gather_time - start; 
    total_time = MPI_Wtime() - start;

	if (IDENT) {
		bool success = true;
		bool global_success;
		for (int i = 0; i < ydim; i++) {
			success && (ylocal[i] == xlocal[i]);
		}
		MPI_Reduce(&success, &global_success, 1, MPI_C_BOOL, MPI_LAND, 0, MPI_COMM_WORLD);
		if (success && !rank && DEBUG) 
			cout << "all processes satisfy ylocal == xlocal assertion" << endl;
   	}

	// Print results for debugging
    if(DEBUG) {
        cout << "Proc (" << ranki << "," << rankj << "), [" << rank << "] started with x values\n";
        for(int j = 0; j < xdim; j++) {
            cout << xlocal[j] << " ";
        }
        cout << "\nand ended with y values\n";
        for(int i = 0; i < ydim; i++) {
            cout << ylocal[i] << " ";
        }
		cout << "\nresult of local matvecmul was\n";
		for (int i = 0; i < mloc; i++) {
			cout << matvecmul_result[i] << " ";
		}
		cout << "\nand had these values in it's recv buffer\n";
		for(int i = 0; i < xdim*pr; i++) {
			cout << recv_buffer[i] << " ";
		}
		cout << "\nand had this part of the input matrix: \n";
		for (int i = 0; i < mloc; i++) {
			for (int j = 0; j < nloc; j++) {
				cout << Alocal[i + j*mloc] << " ";
			}
			cout << "\n";
		}
		cout << "\n";
        cout << endl; // flush now
    }

    // Print time
    if(!rank) {
        cout << nProcs << ","
			<< pr << "," 
			<< pc << "," 
			<< m << "," 
			<< n << "," 
			<< total_time << "," 
			<< gather_time << "," 
			<< comp_time << "," 
			<< reduce_time << "," 
			<< redist_time << endl;
    }

    // Clean up
    delete [] ylocal;
    delete [] xlocal;
    delete [] Alocal;
	delete [] recv_buffer;
	delete [] revcounts;
	delete [] matvecmul_result;
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
}

void local_gemv(double* a, double* x, double* y, int m, int n) {
    // order for loops to match col-major storage
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            y[i] += a[i+j*m] * x[j];
        }
    }
}

void init_rand(double* a, int m, int n, double* x, int xn) {
    // init matrix
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            a[i+j*m] = rand() % 100;
        }
    }
    // init input vector x
    for(int j = 0; j < xn; j++) {
        x[j] = rand() % 100;
    }
}
