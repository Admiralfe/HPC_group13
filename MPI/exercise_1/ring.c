#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	int rank, num_ranks, recv_buff;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

	int next_rank = (rank + 1) % num_ranks;
	int prev_rank = (rank - 1 + num_ranks) % num_ranks;

	MPI_Sendrecv(&rank, 1, MPI_INT, next_rank, 0, &recv_buff, 1, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	printf("Rank %d received %d from rank %d\n", rank, recv_buff, prev_rank);	
	MPI_Finalize();
}

