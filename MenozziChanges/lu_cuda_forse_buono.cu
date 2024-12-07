#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 1024. */
#include "lu.h"
#define BLOCK_SIZE 1024
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
/* CUDA kernel per il calcolo del ciclo su `j` */
#define ARRAY_SIZE 1024;
/*
__global__ void lu_division(DATA_TYPE *A, int n, int k) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j > k && j < n) {
        A[k * n + j] = A[k * n + j] / A[k * n + k]; 
    }
}*/


/*__global__ void lu_elimination(DATA_TYPE *A, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + k + 1;

    if (i < n && j < n && i > k && j > k) {
        A[i * n + j] -= A[i * n + k] * A[k * n + j];
    }
}*/

/*__global__ void reduce(DATA_TYPE *a, int size, int k) {
    // ID globale del thread
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Calcola la riga associata al thread
    int row = k + 1 + blockIdx.y * 1024 + threadIdx.x;  // Adatta la riga in base alla posizione della griglia (prima o seconda riga)
    int col = k + 1 + tid; // Colonna associata al thread

    // Verifica che il thread operi su una colonna valida
    if (col >= size) return;

    // Gestire la parte della matrice in base alla posizione nella griglia
    if (row < size) {
        a[row * size + col] -= a[row * size + k] * a[k * size + col];
    }
}*/

////////////////////////////////////////////////////////////////////
__global__ void lu_division(DATA_TYPE *A, int n, int k) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    // Calcola i valori della riga corrente (k)
    if (j > k && j < n) {
        A[k * n + j] /= A[k * n + k]; // Divisione elemento per il pivot
    }
}

__global__ void lu_elimination(DATA_TYPE *A, int n, int k) {
    int i = threadIdx.y + blockIdx.y * blockDim.y + k + 1;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    // Calcola i valori delle righe successive (i > k)
    if (i < n && j > k && j < n) {
        A[i * n + j] -= A[i * n + k] * A[k * n + j];
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

    for (int k = 0; k < n; k++) {
        // Divisione: Calcolo della riga k
        lu_division<<<gridDim, threadsPerBlockDim>>>(d_A, n, k);

        // Sincronizzazione per garantire che la riga k sia completata
        cudaDeviceSynchronize();

        // Eliminazione: Aggiornamento delle righe successive
        dim3 threadsPerBlock2D(16, 16);  // Configurazione 2D per thread
        dim3 gridDim2D((n + 15) / 16, (n + 15) / 16);
        lu_elimination<<<gridDim2D, threadsPerBlock2D>>>(d_A, n, k);

        // Sincronizzazione per garantire che la fase di eliminazione sia completata
        cudaDeviceSynchronize();
    }

    // Copia del risultato sulla CPU
    cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);

    // Liberazione della memoria sulla GPU
    cudaFree(d_A);
}
////////////////////////////////////////////////////////////////////
/*
__global__ void reduce(DATA_TYPE *a, int size, int k) {
    // Global thread ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Mapping the column associated with this thread
    int col = k + 1 + tid;

    // Ensure the column index is within bounds
    if (col >= size) return;

    // Iterate over rows starting from k+1 (skip the pivot row)
    for (int row = k + 1 + blockIdx.y * BLOCK_SIZE; row < size; row++) {
        // Perform LU elimination for this specific row and column
        a[row * size + col] -= a[row * size + k] * a[k * size + col];
    }
}

static void kernel_lu(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n)) {
    DATA_TYPE *d_A;
    size_t size = n * n * sizeof(DATA_TYPE);

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    // Configuration for grid and threads
    int threadsPerBlock = 1024;  // Up to 1024 threads per block
    dim3 threadsPerBlockDim(threadsPerBlock);  // Number of threads per block

    for (int k = 0; k < n; k++) {
        // Division step (computing the k-th row)
        int blocks1D = (n - k + threadsPerBlock - 1) / threadsPerBlock;
        lu_division<<<blocks1D, threadsPerBlockDim>>>(d_A, n, k);

        // Perform the elimination step, dividing the matrix into blocks
        int blocksPerGrid_y = (n - k + BLOCK_SIZE - 1) / BLOCK_SIZE;  // Calculate the number of blocks needed
        dim3 blocksPerGrid(blocksPerGrid_y);

        // Execute the kernel to update the matrix (LU elimination)
        reduce<<<blocksPerGrid, threadsPerBlockDim>>>(d_A, n, k);
    }

    // Copy the results back from GPU to CPU
    cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(d_A);
}
*/


    


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
            if (abs(A_serial[i][j] - A_cuda[i][j]) > 0.0001) {
                dead_counter++;
                printf("Differenza trovata in A[%d][%d]: serial=%.16f, cuda=%.16f\n", i, j, A_serial[i][j], A_cuda[i][j]);
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
    //printf("N è : %d \n",n);  
    /* Variable declaration/allocation. */
    POLYBENCH_2D_ARRAY_DECL(A_cuda, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(A_serial, DATA_TYPE, N, N, n, n);
    /* Initialize array(s). */
    init_array(n, POLYBENCH_ARRAY(A_cuda));
    init_array(n, POLYBENCH_ARRAY(A_serial));

    /* Start timer. */
    printf("Cuda: ");
    polybench_start_instruments;

    /* Run kernel. */
    kernel_lu(n, POLYBENCH_ARRAY(A_cuda));

    /* Stop and print timer. */
    polybench_stop_instruments;
    polybench_print_instruments;
    
    printf("Seriale: ");
    polybench_start_instruments;
    kernel_lu_serial(n, POLYBENCH_ARRAY(A_serial));
    polybench_stop_instruments;
    polybench_print_instruments;

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
