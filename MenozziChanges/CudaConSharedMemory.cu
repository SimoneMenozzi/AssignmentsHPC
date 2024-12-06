#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 1024. */
#include "lu.h"
#define BLOCK_SIZE 32
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


#define BLOCK_SIZE 32 // Tune block size based on GPU architecture

__global__ void lu_division(DATA_TYPE *A, int n, int k) {
    int j = threadIdx.x + blockIdx.x * blockDim.x + k +1;

    if (j > k && j < n) {
        A[k * n + j] = A[k * n + j] / A[k * n + k];
    }
}

__global__ void lu_elimination(DATA_TYPE *A, int n, int k) {
    // Calculate the row and column indices for each thread in the 2D grid
    int tx = threadIdx.x; // Thread index within the block (x direction)
    int ty = threadIdx.y; // Thread index within the block (y direction)

    // Calculate the global row and column indices
    int row = blockIdx.x * blockDim.x + tx + k + 1; // Global row index
    int col = blockIdx.y * blockDim.y + ty + k + 1; // Global column index

    // Ensure the thread is working within bounds
    if (row < n && col < n) {
        // Shared memory to store the pivot row and pivot column
        __shared__ DATA_TYPE pivot_row[BLOCK_SIZE];
        __shared__ DATA_TYPE pivot_col[BLOCK_SIZE];

        // Load the pivot row and column into shared memory
        if (tx == 0 && col < n) {
            pivot_row[ty] = A[k * n + col];  // Load the pivot row into shared memory
        }
        if (ty == 0 && row < n) {
            pivot_col[tx] = A[row * n + k];  // Load the pivot column into shared memory
        }

        __syncthreads();  // Synchronize threads to ensure shared memory is loaded

        // Perform the elimination
        if (row > k && col > k) {
            A[row * n + col] -= pivot_col[tx] * pivot_row[ty];
        }
    }
}


static void kernel_lu(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n)) {
    DATA_TYPE *d_A;
    size_t size = n * n * sizeof(DATA_TYPE);

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    // Thread and block configurations
    int threadsPerBlock1D = 256; // For division
    dim3 threadsPerBlock2D(32, 32); // For elimination
    dim3 gridPerElim((n + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,
                     (n + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y);
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE); // Each block is BLOCK_SIZE x BLOCK_SIZE
    dim3 blocksPerGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE); // 2D grid to cover the matrix

    for (int k = 0; k < n; k++) {
        // Division step
        int blocks1D = (n - k + threadsPerBlock1D - 1) / threadsPerBlock1D;
        lu_division<<<blocks1D, threadsPerBlock1D>>>(d_A, n, k);

        


        lu_elimination<<<blocksPerGrid, threadsPerBlock>>>(d_A, n, k);


    }

    // Copy back the results
    cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);
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
