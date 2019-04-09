#include "sum.h"

void omp_sum(double *sum_ret)
{
	int num_threads = omp_get_max_threads();
	double[num_threads] partial_sums;
#pragma omp parallel
{
	int id = omp_get_thread_num();
	for (int i = id; i < size; i+= num_threads){
		partial_sums[i] += x[i];
	}
}
	int sum = 0;
	for (int i = 0; i < num_threads; i++) {
		sum += partial_sums[i]
	}
	
	*sum_ret = sum;
}

void omp_critical_sum(double *sum_ret)
{

}

void omp_atomic_sum(double *sum_ret)
{

}

void omp_local_sum(double *sum_ret)
{

}

void omp_padded_sum(double *sum_ret)
{

}

void omp_private_sum(double *sum_ret)
{

}

void omp_reduction_sum(double *sum_ret)
{

}
