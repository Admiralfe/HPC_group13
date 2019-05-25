#include "block_matmul.h"

struct Config {
	/* MPI Files */
	MPI_File A_file, B_file, C_file;
	char *outfile;

	/* MPI Datatypes for matrix blocks */
	MPI_Datatype block;

	/* Matrix data */
	double *A, *A_tmp, *B, *C;

	/* Cart communicators */
	MPI_Comm grid_comm;
	MPI_Comm row_comm;
	MPI_Comm col_comm;

	/* Cart communicator dim and ranks */
	int dim[2], coords[2];
	int world_rank, world_size, grid_rank;
	int row_rank, row_size, col_rank, col_size;

	/* Full matrix dim */
	int A_dims[2];
	int B_dims[2];
	int C_dims[2];
	int matrix_size;

	/* Process local matrix dim */
	int local_dims[2];
	int local_size;
};

struct Config config;

void multiply_matrices_and_add(double *A, double *B, double *C)
{
	int i, j, k ;
	int n = config.local_dims[0];
	// Loop nest optimized algorithm
	for (i = 0 ; i < n; i++)
 		for (k = 0 ; k < n ; k++)
  			for (j = 0 ; j < n ; j++)
  				C[i*n + j] += A[i*n + k] * B[k*n + j];
}

void init_matmul(char *A_file, char *B_file, char *outfile)
{
	MPI_Comm_rank(MPI_COMM_WORLD, &config.world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &config.world_size);
	/* Copy output file name to configuration */
	config.outfile = outfile;
	/* Get matrix size header */
	MPI_File_open(MPI_COMM_WORLD, A_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &config.A_file);
	MPI_File_open(MPI_COMM_WORLD, B_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &config.B_file);
	if (config.world_rank == 0)
	{
		MPI_File_read_at(config.A_file, 0, config.A_dims, 2, MPI_INT, MPI_STATUS_IGNORE);
		MPI_File_read_at(config.B_file, 0, config.B_dims, 2, MPI_INT, MPI_STATUS_IGNORE);
        }
	MPI_Bcast(&config.A_dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&config.B_dims, 2, MPI_INT, 0, MPI_COMM_WORLD);

	/* Set dim of tiles relative to the number of processes as NxN where N=sqrt(world_size) */
	config.dim[0] = (int) sqrt(config.world_size);
	config.dim[1] = (int) sqrt(config.world_size);

	/* Verify dim of A and B matches for matul and both are square*/
	assert(config.A_dims[1] == config.B_dims[0]);
	assert(config.A_dims[0] == config.A_dims[1]);
	assert(config.B_dims[0] == config.B_dims[1]);

	config.local_dims[0] = config.A_dims[0] / config.dim[0];
	config.local_dims[1] = config.A_dims[1] / config.dim[1];
	config.C_dims[0] = config.A_dims[0];
	config.C_dims[1] = config.B_dims[1];

	/* Create Cart communicator for NxN processes */
	int wrap_around[2] = {1, 0};
	MPI_Cart_create(MPI_COMM_WORLD, 2, config.dim, wrap_around, 1, &config.grid_comm);
	MPI_Comm_rank(config.grid_comm, &config.grid_rank);

	/* Sub div cart communicator to N row communicator */
	int keep_dims[2] = {0, 1};
	MPI_Cart_sub(config.grid_comm, keep_dims, &config.row_comm);
	MPI_Comm_rank(config.row_comm, &config.row_rank);
	MPI_Comm_size(config.row_comm, &config.row_size);

	/* Sub div cart communicator to N col communicator */
	keep_dims[0] = 1;
	keep_dims[1] = 0;
	MPI_Cart_sub(config.grid_comm, keep_dims, &config.col_comm);
	MPI_Comm_rank(config.col_comm, &config.col_rank);
	MPI_Comm_size(config.col_comm, &config.col_size);

	/* Setup sizes of full matrices */
	config.matrix_size = config.A_dims[0] * config.A_dims[1];
	/* Setup sizes of local matrix tiles */
	config.local_size = config.local_dims[0] * config.local_dims[1];
	/* Create subarray datatype for local matrix tile */
	int starts[2];
	starts[0] = config.local_dims[0] * config.col_rank;
	starts[1] = config.local_dims[1] * config.row_rank;
	MPI_Type_create_subarray(2, config.A_dims, config.local_dims, starts, MPI_ORDER_C, MPI_DOUBLE, &config.block);
	MPI_Type_commit(&config.block);

	/* Create data array to load actual block matrix data */
	config.A = (double*) malloc(config.local_size * sizeof(double));
	config.B = (double*) malloc(config.local_size * sizeof(double));
	config.C = (double*) calloc(config.local_size, sizeof(double));

	/* Set fileview of process to respective matrix block */
	MPI_Offset header_displacement = 2 * sizeof(int);
	MPI_File_set_view(config.A_file, header_displacement, MPI_DOUBLE, config.block, "native", MPI_INFO_NULL);
	MPI_File_set_view(config.B_file, header_displacement, MPI_DOUBLE, config.block, "native", MPI_INFO_NULL);

	/* Collective read blocks from files */
	MPI_File_read_all(config.A_file, config.A, config.local_size, MPI_DOUBLE, MPI_STATUS_IGNORE);
	MPI_File_read_all(config.B_file, config.B, config.local_size, MPI_DOUBLE, MPI_STATUS_IGNORE);
	
	/* Close data source files */
	MPI_File_close(&config.A_file);
	MPI_File_close(&config.B_file);
}

void cleanup_matmul()
{
	MPI_File_open(MPI_COMM_WORLD, config.outfile, MPI_MODE_WRONLY, MPI_INFO_NULL, &config.C_file);
	/* Rank zero writes header specifying dim of result matrix C */
	if (config.row_rank == 0)
		MPI_File_write_at(config.C_file, 0, config.C_dims, 2, MPI_INT, MPI_STATUS_IGNORE);
	/* Set fileview of process to respective matrix block with header offset */
	MPI_Offset header_displacement = 2 * sizeof(int);
	MPI_File_set_view(config.C_file, header_displacement, MPI_DOUBLE, config.block, "native", MPI_INFO_NULL);
	/* Collective write and close file */
	MPI_File_write_all(config.C_file, config.C, config.local_size, MPI_DOUBLE, MPI_STATUS_IGNORE);
	/* Cleanup */
	free(config.A);
	free(config.B);
	free(config.C);
	MPI_File_close(&config.C_file);
}

void compute_fox()
{
	/* Compute source and target for verticle shift of B blocks */
	int source, target;
	int bcast_root_rank; //Rank of the broadcasting process in the row communicators
	MPI_Cart_shift(config.col_comm, 0, -1, &source, &target);
	config.A_tmp = (double*) malloc(config.matrix_size * sizeof(double));
	memset(config.C, 0, config.local_size * sizeof(double));
	for (int i = 0; i < config.dim[0]; i++) {
		/* Diag + i broadcast block A horizontally and use A_tmp to preserve own local A */
		bcast_root_rank = (config.col_rank + i) % config.dim[0]; //Diagonal has rank same as column rank of current row.
		if (config.row_rank == bcast_root_rank)
			memcpy(config.A_tmp, config.A, sizeof(double) * config.local_size);
		MPI_Bcast(config.A_tmp, config.matrix_size, MPI_DOUBLE, bcast_root_rank, config.row_comm);
		/* dgemm with blocks */
		multiply_matrices_and_add(config.A_tmp, config.B, config.C);
		/* Shfting block B upwards and receive from process below */
		MPI_Sendrecv_replace(config.B, config.local_size, MPI_DOUBLE, target, 0, source, 0, config.col_comm, MPI_STATUS_IGNORE);
	}
	free(config.A_tmp);
}
