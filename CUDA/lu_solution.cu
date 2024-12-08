#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 1024. */
#include "lu.h"
#define BLOCK_SIZE 16
/* Array initialization. */
static void init_array(int n,
                       DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
    int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                A[i][j] = ((DATA_TYPE)(i + 1) * (j + 1)) / n + 1.0; // Aggiungi offset di 1
            } else {
                A[i][j] = ((DATA_TYPE)(i + 1) * (j + 1)) / n;
            }
        }
    }
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


__global__ void lu_division(DATA_TYPE *A, int n, int k) {
    int j = threadIdx.x + blockIdx.x * blockDim.x + k + 1;

    // Calcola i valori della riga corrente (k)
    if (j > k && j < n) {
        A[k * n + j] /= A[k * n + k]; // Divisione elemento per il pivot
    }
}

__global__ void lu_elimination(DATA_TYPE *A, int n, int k) {
    // Thread and block IDs
    int tx = threadIdx.x;  // Local thread ID within the block (x)
    int ty = threadIdx.y;  // Local thread ID within the block (y)
    int bx = blockIdx.x;   // Block ID along columns
    int by = blockIdx.y;   // Block ID along rows

    // Calculate global row and column indices
    int i = by * blockDim.y + ty + k + 1; // Rows start after pivot row k
    int j = bx * blockDim.x + tx + k + 1; // Columns start after pivot column k

    // Declare shared memory for the pivot row and column
    __shared__ DATA_TYPE pivot_row[BLOCK_SIZE];
    __shared__ DATA_TYPE pivot_col[BLOCK_SIZE];

    // Load pivot data into shared memory
    if (ty == 0 && j < n) {  // One thread per column loads pivot row
        pivot_row[tx] = A[k * n + j];
    }
    if (tx == 0 && i < n) {  // One thread per row loads pivot column
        pivot_col[ty] = A[i * n + k];
    }

    // Synchronize threads to ensure shared memory is populated
    __syncthreads();

    // Perform elimination for the assigned element
    if (i < n && j < n) {
        A[i * n + j] -= pivot_col[ty] * pivot_row[tx];
    }
}



static void kernel_lu(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n)) {
    DATA_TYPE *d_A;
    size_t size = n * n * sizeof(DATA_TYPE);

    // Allocazione sulla GPU
    cudaMalloc((void **)&d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    // Configurazione dei thread e dei blocchi
    int threadsPerBlock = 256;  // Numero massimo di thread per blocco
    dim3 threadsPerBlockDim(threadsPerBlock);
    dim3 gridDim((n + threadsPerBlock - 1) / threadsPerBlock);
     dim3 threadsPerBlock2D(BLOCK_SIZE, BLOCK_SIZE);
    for (int k = 0; k < n; k++) {
        // Divisione: Calcolo della riga k
        lu_division<<<gridDim, threadsPerBlockDim>>>(d_A, n, k);

        // Sincronizzazione per garantire che la riga k sia completata
 
        // Eliminazione: Aggiornamento delle righe successive  // Configurazione 2D per thread
        dim3 gridDim2D((n - k +  BLOCK_SIZE - 1) / BLOCK_SIZE, (n - k + BLOCK_SIZE - 1) / BLOCK_SIZE);
        lu_elimination<<<gridDim2D, threadsPerBlock2D>>>(d_A, n, k);

        // Sincronizzazione per garantire che la fase di eliminazione sia completata
    }

    // Copia del risultato sulla CPU
    cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);

    // Liberazione della memoria sulla GPU
    cudaFree(d_A);
}



 // Eliminate step
       /* int remaining_cols = n - k - 1; // Colonne rimanenti
        int threads_per_block = 1024; // Massimo 1024 thread per blocco
        int blocks_needed = (remaining_cols + threads_per_block - 1) / threads_per_block;*/

       
        /*int grid_dim_x = (n - k + 1023) / 1024;  
        int grid_dim_y = (n - k  + 1023) / 1024;  

        
        dim3 blocksPerGrid(grid_dim_x, grid_dim_y);
        dim3 threadsPerBlock(threads_per_block);

        reduce<<<blocksPerGrid, threadsPerBlock>>>(d_A, n, k);*/

void kernel_lu_serial(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n)) {
    int i,j,k;
   /*printf("seriale\n");
    for (int i = 0; i < _PB_N; i++) {
        for (int j = 0; j < _PB_N; j++) {
            printf("%.6f ", A[i][j]);
        }
        printf("\n");
    }
*/

    for (k = 0; k < _PB_N; k++) {
	    for (j = k + 1; j < _PB_N; j++) {
            A[k][j] =A[k][j] / A[k][k];
	    }    
        
        for (i = k + 1; i < _PB_N; i++) { 	
	        for (j = k + 1; j < _PB_N; j++) {
		        
                A[i][j] -= A[i][k] * A[k][j];
            }
        
	    }
    }
    /*printf("seriale dopo tutto \n"); 
    for (int i = 0; i < _PB_N; i++) {
        for (int j = 0; j < _PB_N; j++) {
            printf("%.6f ", A[i][j]);
        }
        printf("\n");
    }*/
}

void test_correctness(int n, DATA_TYPE POLYBENCH_2D(A_serial, N, N, n, n), DATA_TYPE POLYBENCH_2D(A_cuda, N, N, n, n)) {
    int dead_counter = 0;
    printf("Confronto risultati:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (A_serial[i][j] - A_cuda[i][j] != 0) {
                dead_counter++;
                printf("Differenza trovata in A[%d][%d]: serial=%.6f, cuda=%.6f\n", i, j, A_serial[i][j], A_cuda[i][j]);
                if(dead_counter == 100)
                    break;
            }
        }
        if(dead_counter == 100)
        break;
    }
    return;
    printf("I risultati sono equivalenti.\n");
}
 
 
int main(int argc, char **argv) {
    /* Retrieve problem size. */
    //int n = N;
    int n = N;
    printf("N è : %d \n",n);  
    /* Variable declaration/allocation. */
    POLYBENCH_2D_ARRAY_DECL(A_cuda, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(A_serial, DATA_TYPE, N, N, n, n);
    /* Initialize array(s). */
    init_array(n, POLYBENCH_ARRAY(A_cuda));
    init_array(n, POLYBENCH_ARRAY(A_serial));

    /* Start timer. */
    polybench_start_instruments;

    /* Run kernel. */
    kernel_lu(n, POLYBENCH_ARRAY(A_cuda));

    /* Stop and print timer. */
    polybench_stop_instruments;
    polybench_print_instruments;
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
