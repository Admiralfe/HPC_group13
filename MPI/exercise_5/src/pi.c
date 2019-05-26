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
	
	//Create the output char array
	int out_str_buffer_len = 20;
	char out_str[out_str_buffer_len];
	//Set all characters in the array to blankspace character, aka ASCII 32
	for (i = 0; i < out_str_buffer_len; i++)
		out_str[i] = 32;
	
	int written_chars = 
		snprintf(out_str, out_str_buffer_len, "%d %.6f", rank, *local_count / (double) flip);
	out_str[written_chars] = 32; //Remove the null terminator since we will write to file as binary
	out_str[out_str_buffer_len - 1] = 10; //Manually add new line character. (ASCII 10)

	//Write to the output file
	MPI_File fh;
	MPI_File_open(MPI_COMM_WORLD, "results.txt", MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
	MPI_File_write_at(fh, rank * out_str_buffer_len, out_str, out_str_buffer_len, MPI_CHAR, MPI_STATUS_IGNORE);

	int count;
        MPI_Reduce(local_count, &count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
                pi = ((double) count / (double) (flip * num_ranks)) * 4.0;
                *answer = pi;
		written_chars = snprintf(out_str, out_str_buffer_len, "pi = %.6f", pi);
		out_str[written_chars] = 32; //remove the null terminator since we are writing in binary.
		out_str[out_str_buffer_len - 1] = 10; //Manually add new line character (ASCII 10)
		MPI_File_write_at(fh, out_str_buffer_len * num_ranks, out_str, out_str_buffer_len, MPI_CHAR, MPI_STATUS_IGNORE);
        }
        return;
}

