#include "pi.h"

#define SEED 921
#define NUM_ITER 1000000000

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
	int rank, num_ranks;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
		
    double x, y, z, pi;
    
    srand(SEED * rank); // Important: Multiply SEED by "rank" when you introduce MPI!
    
    // Calculate PI following a Monte Carlo method
    for (int iter = 0; iter < local_iters; iter++)
    {
        // Generate random (X,Y) points
        x = (double)random() / (double)RAND_MAX;
        y = (double)random() / (double)RAND_MAX;
        z = sqrt((x*x) + (y*y));
        
        // Check if point is in unit circle
        if (z <= 1.0)
        {
            (*local_count)++;
        }
    }
    
	if (rank == 0) 
	{
		int counts[num_ranks - 1];
		MPI_Request requests[num_ranks - 1];
		int count = 0;
		
		for (int i = 1; i < num_ranks; i++)
		{
			MPI_Irecv(&counts[i - 1], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
		}
		MPI_Waitall(num_ranks - 1, requests, MPI_STATUSES_IGNORE);
		
		for (int i = 0; i < num_ranks; i++) 
		{
			count += counts[i];
		}
		// Estimate Pi and display the result
		pi = ((double)count / (double)NUM_ITER) * 4.0;
		*answer = pi;
	} 
	else 
	{
		MPI_Send(local_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);	
	}
	
	return;
}
