#include "sum.h"

void omp_sum(double *sum_ret)
{
        double sum = 0;
#pragma omp parallel
{
	int num_threads = omp_get_num_threads();
	int id = omp_get_thread_num();
        for (int i = id; i < size; i+=num_threads){
                sum += x[i];
        }
}
        *sum_ret = sum;
}


void omp_critical_sum(double *sum_ret)
{
        double sum = 0;
#pragma omp parallel
{
        int num_threads = omp_get_num_threads();
        int id = omp_get_thread_num();
        for (int i = id; i < size; i+=num_threads){
		#pragma omp critical
                sum += x[i];
        }
}
        *sum_ret = sum;
}


void omp_atomic_sum(double *sum_ret)
{
        double sum = 0;
#pragma omp parallel
{
        int num_threads = omp_get_num_threads();
        int id = omp_get_thread_num();
        for (int i = id; i < size; i+=num_threads){
                #pragma omp atomic
                sum += x[i];
        }
}
        *sum_ret = sum;
}


void omp_local_sum(double *sum_ret)
{
        int num_threads = omp_get_max_threads();
        double* partial_sums = calloc(num_threads, sizeof(double));
#pragma omp parallel
{
        int local_num_threads = omp_get_num_threads();
        int id = omp_get_thread_num();
        for (int i = id; i < size; i+= local_num_threads){
                partial_sums[id] += x[i];
        }
}
        double sum = 0;
        for (int i = 0; i < num_threads; i++) {
                sum += partial_sums[i];
        }

        *sum_ret = sum;
        free(partial_sums);
}

void omp_padded_sum(double *sum_ret)
{
	int pad = 16;
        int num_threads = omp_get_max_threads();
        double* partial_sums = calloc(num_threads, pad * sizeof(double));
#pragma omp parallel
{
        int local_num_threads = omp_get_num_threads();
        int id = omp_get_thread_num();
        for (int i = id; i < size; i+= local_num_threads){
               partial_sums[pad * id] += x[i];
        }
}
        double sum = 0;
        for (int i = 0; i < num_threads; i++) {
                sum += partial_sums[pad * i];
        }

        *sum_ret = sum;
        free(partial_sums);
}


void omp_private_sum(double *sum_ret)
{
        double local_sum;
	double global_sum = 0;
#pragma omp parallel private(local_sum)
{
	local_sum = 0;
        int local_num_threads = omp_get_num_threads();
        int id = omp_get_thread_num();
        for (int i = id; i < size; i+= local_num_threads){
                local_sum += x[i];
        }
	#pragma omp atomic
	global_sum += local_sum;
}
        *sum_ret = global_sum;
}


void omp_reduction_sum(double *sum_ret)
{
        double sum = 0;
	#pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < size; i++){
                sum += x[i];
        }
        *sum_ret = sum;
}

