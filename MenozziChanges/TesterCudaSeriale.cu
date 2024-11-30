#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 1024. */
#include "lu.h"

/* Array initialization. */
static void init_array(int n,
                       DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A[i][j] = ((DATA_TYPE)(i + 1) * (j + 1)) / n;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n,
                        DATA_TYPE POLYBENCH_2D(A, N, N, n, n))

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
      if ((i * n + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/*
#pragma omp declare target
static void kernel_lu(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
    int i, j, k;
    #pragma omp target data map(tofrom: A[0:n][0:n])
    {
        #pragma omp target teams num_teams(_PB_N/NTHREADS_GPU) thread_limit(NTHREADS_GPU)
        {
            for (k = 0; k < _PB_N; k++) {
              
                #pragma omp distribute parallel for simd num_threads(NTHREADS_GPU) schedule(static,NTHREADS_GPU)
                for (j = k + 1; j < _PB_N; j++) {
                    A[k][j] = A[k][j] / A[k][k];
                }

                #pragma omp distribute parallel for simd  num_threads(NTHREADS_GPU) schedule(dynamic,NTHREADS_GPU) 
                for (i = k + 1; i < _PB_N; i++) {
                    for (j = k + 1; j < _PB_N; j++) {
                        A[i][j] -= A[i][k] * A[k][j];
                    } 
                }
            }
        }
    }
}
#pragma omp end declare target
*/
/*
static void kernel_lu(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
    int i, j, k;
     #pragma omp target data map(tofrom: A[0:n][0:n])
    {
            
            for (k = 0; k < _PB_N; k++) {
                #pragma omp  parallel for simd num_threads(NTHREADS_GPU) schedule(static,NTHREADS_GPU)
                for (j = k + 1; j < _PB_N; j++) {
                    A[k][j] = A[k][j] /A[k][k];;
                }

                #pragma omp  parallel for simd  num_threads(NTHREADS_GPU) schedule(dynamic,NTHREADS_GPU) 
                for (i = k + 1; i < _PB_N; i++) {
                    for (j = k + 1; j < _PB_N; j++) {
                        A[i][j] -=  A[i][k] * A[k][j];
                    } 
                }
            }
      }
}
*/
/*
static void kernel_lu(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
    int i, j, k;
            for (k = 0; k < _PB_N; k++) {
               
                for (j = k + 1; j < _PB_N; j++) {
                    A[k][j] = A[k][j] / A[k][k];
                }
                for (i = k + 1; i < _PB_N; i++) {
                    for (j = k + 1; j < _PB_N; j++) {
                        A[i][j] -= A[i][k] * A[k][j];
                    } 
                }
            }
}
*/
/* CUDA kernel per il calcolo del ciclo su `j` */

__global__ void lu_division(DATA_TYPE *A, int n, int k) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j > k && j < n) {
        A[k * n + j] = A[k * n + j] / A[k * n + k]; 
    }
}

/* CUDA kernel per il calcolo del ciclo su `i` e `j` */
__global__ void lu_elimination(DATA_TYPE *A, int n, int k) {
    int i = blockIdx.x + k + 1; // elemento sotto alla diagonale di inizio
    int j = threadIdx.x + blockIdx.y * blockDim.y;

    if (i < n && j > k && j < n) {
        A[i * n + j] -= A[i * n + k] * A[k * n + j];
    }
}

static void kernel_lu(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n)) {
    DATA_TYPE *d_A;
    size_t size = n * n * sizeof(DATA_TYPE);

    /* Allocazione memoria sulla GPU */
    cudaMalloc((void **)&d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(128); // valore statico da controllare

    for (int k = 0; k < n; k++) {
        /* Calcolo del ciclo su `j` */
        dim3 blocksPerGrid((n - k + threadsPerBlock.x - 1) / threadsPerBlock.x); //-1 per utilizzare meglio la gestione delle matrici in griglia
        lu_division<<<blocksPerGrid, threadsPerBlock>>>(d_A, n, k);

        /* Sincronizzazione per evitare dipendenze */
        cudaDeviceSynchronize(); //Differenza tra __syncthread() è che questo sincronizza tutta la gpu, l'altro blocco per blocco

        /* Calcolo del ciclo su `i` e `j` */
        dim3 threadsPerBlock2D(16, 16);// valore statico da cambiare
        dim3 blocksPerGrid2D((n - k + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,
                             (n - k + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y);
        lu_elimination<<<blocksPerGrid2D, threadsPerBlock2D>>>(d_A, n, k);

        /* Sincronizzazione */
        cudaDeviceSynchronize();
    }
 
    /* Copia i risultati dalla GPU alla CPU */
    cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);

    /* Libera la memoria sulla GPU */
    cudaFree(d_A);
}
void kernel_lu_serial(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n)) {
    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
        }
    }
}

void test_correctness(int n, DATA_TYPE POLYBENCH_2D(A_serial, N, N, n, n), DATA_TYPE POLYBENCH_2D(A_cuda, N, N, n, n)) {
    printf("Confronto risultati:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(A_serial[i][j] - A_cuda[i][j]) > 1) {
                printf("Differenza trovata in A[%d][%d]: serial=%.6f, cuda=%.6f\n", i, j, A_serial[i][j], A_cuda[i][j]);
                return;
            }
        }
    }
    printf("I risultati sono equivalenti.\n");
}


int main(int argc, char **argv) {
    /* Retrieve problem size. */
    int n = N;

    /* Variable declaration/allocation. */
    POLYBENCH_2D_ARRAY_DECL(A_cuda, DATA_TYPE, N, N, n, n);

    /* Initialize array(s). */
    init_array(n, POLYBENCH_ARRAY(A_cuda));

    /* Start timer. */
    polybench_start_instruments;

    /* Run kernel. */
    kernel_lu(n, POLYBENCH_ARRAY(A_cuda));

    /* Stop and print timer. */
    polybench_stop_instruments;
    polybench_print_instruments;

    POLYBENCH_2D_ARRAY_DECL(A_serial, DATA_TYPE, N, N, n, n);
    init_array(n, POLYBENCH_ARRAY(A_serial));
    kernel_lu_serial(n, POLYBENCH_ARRAY(A_serial));

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A_serial)));
    polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A_cuda)));

    test_correctness(n, POLYBENCH_ARRAY(A_serial), POLYBENCH_ARRAY(A_cuda));

    /* Be clean. */
    POLYBENCH_FREE_ARRAY(A_serial);
    POLYBENCH_FREE_ARRAY(A_cuda);


    return 0;
}
