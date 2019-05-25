#include "pi.h"
#define SEED 921

void init_pi(int set_seed, char *outfile)
{
	if (filename != NULL) {
		free(filename);
		filename = NULL;
	}

	if (outfile != NULL) {
		filename = (char*)calloc(sizeof(char), strlen(outfile)+1);
		memcpy(filename, outfile, strlen(outfile));
		filename[strlen(outfile)] = 0;
	}
	seed = set_seed;
}

void cleanup_pi()
{
	if (filename != NULL)
		free(filename);
}

void compute_pi(int flip, int *local_count, double *answer)
{
	int rank, num_ranks, i;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
		
	double x, y, z, pi;
    
	srand(SEED * rank); // Important: Multiply SEED by "rank" when you introduce MPI!
    
	// Calculate PI following a Monte Carlo method
	for (i = 0; i < flip; i++)
	{
		// Generate random (X,Y) points
		x = (double)random() / (double)RAND_MAX;
		y = (double)random() / (double)RAND_MAX;
		z = sqrt((x*x) + (y*y));
        
		// Check if point is in unit circle
		if (z <= 1.0)
			(*local_count)++;
	}
	int count;
	MPI_Reduce(local_count, &count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		pi = ((double) count / (double) (flip * num_ranks)) * 4.0;
		*answer = pi;
	}
	return;
}
