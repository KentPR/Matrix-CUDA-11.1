#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand_kernel.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <conio.h>

//CPU - host
//GPU - device

//blockDim  - dimention of block
//blockIdx  - index of current block
//threadIdx - index of current thread in block

__device__ void elem(double* A, int m, int n, double kof, int N) //execution on Device
{
    int tid = blockIdx.y * blockDim.y + threadIdx.y;
    if (tid < N) 
        A[m * N + tid] -= kof * A[n * N + tid];
}

__global__ void triangle_kernel(double* A, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int j;
    double kof;
    for (j = 0; j < N * N; j++)
    {
        if (tid < N - 1 && tid >= j)
        {
            kof = A[(tid + 1) * N + j] / A[j * N + j];
            elem(A, tid + 1, j, kof, N);
        }
    }
}


double det(double* arr, int N)
{
    double d = 1.0;
    for (int i = 0; i < N; i++)  d *= arr[i * N + i];
    return d;
}


void print_cuda_device_info(cudaDeviceProp& prop)
{
    printf("Device... ... ...initialized!");
    printf("Device name:                                        %s\n", prop.name);
    //printf("Global memory available on device:                  %zu\n", prop.totalGlobalMem);
    //printf("Shared memory available per block:                  %zu\n", prop.sharedMemPerBlock);
    printf("Warp size in threads:                               %d\n", prop.warpSize);
    printf("Maximum number of threads per block:                %d\n", prop.maxThreadsPerBlock);
    /*
    printf("Maximum size of each dimension of a block[0]:       %d\n", prop.maxThreadsDim[0]);
    printf("Maximum size of each dimension of a block[1]:       %d\n", prop.maxThreadsDim[1]);
    printf("Maximum size of each dimension of a block[2]:       %i\n", prop.maxThreadsDim[2]);
    */
    printf("Maximum size of each dimension of a grid[0]:        %i\n", prop.maxGridSize[0]);
    /*
    printf("Maximum size of each dimension of a grid[1]:        %i\n", prop.maxGridSize[1]);
    printf("Maximum size of each dimension of a grid[2]:        %i\n", prop.maxGridSize[2]);
    printf("Clock frequency in kilohertz:                       %i\n", prop.clockRate);
    printf("totalConstMem:                                      %zu\n", prop.totalConstMem);
    printf("Major compute capability:                           %i\n", prop.major);
    printf("Minor compute capability:                           %i\n", prop.minor);
    */
    printf("Number of multiprocessors on device:                %i\n", prop.multiProcessorCount);
}

__host__ int main()
{
    int N;
    printf("Input size of matrix N = ");
    scanf_s("%i", &N);
    unsigned int timer;


    int Matrix_size = N * N; //Size of matrix
    int MatrixTotalMemory = Matrix_size * sizeof(double);//Память, необходимая для массива на GPU 
    double* InputMatrix = new double[Matrix_size];//Выделяем память под массив

    //Заполняем матрицу случайными числами и выводим на экран
    srand(time(NULL));
    for (int i = 0; i < Matrix_size; i++)
    {
        InputMatrix[i]  = 1 + rand() % 9;
    }

    printf("\n");
    for (int i = 0; i < Matrix_size; i++)
    {
        printf("(%0.0f)", InputMatrix[i]);
        if (((i+1) % N == 0) && (i != 0)) 
            printf("\n");
    }
    printf("\n");
    _getch();


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    //print_cuda_device_info(prop);
    double* MatrixDeviceMemory;
     
    dim3 gridSize = dim3(N, N, 1);  //Dimention of Grid (matrix N*N)
    dim3 blockSize = dim3(1, 1, 1); //Dimention of block 


    //Инициализируем переменные для замера времени работы
    float run_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMalloc((void**)&MatrixDeviceMemory, MatrixTotalMemory); //allocating memory on GPU
    cudaMemcpy(MatrixDeviceMemory, InputMatrix, MatrixTotalMemory, cudaMemcpyHostToDevice); //copying operands to GPU

    //float start2 = clock(); //Fix the begin of work in timeline.

    triangle_kernel <<< gridSize, blockSize >>> (MatrixDeviceMemory, N); //Execution of matrix triangling
    cudaThreadSynchronize();
    cudaEventSynchronize(stop); 

    //float end = clock();   //Fix the end of execution

    cudaMemcpy(InputMatrix, MatrixDeviceMemory, MatrixTotalMemory, cudaMemcpyDeviceToHost);//Копируем новую матрицу с GPU на CPU

    //Получаем время работы
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&run_time, start, stop);

    
    

    //Выводим полученную матрицу
    int string = 0;

    for (int i = 0; i < Matrix_size; i++)
    {
        if (string && i == string * N)
        {
            int m = i;
            for (int j = string * N; j < string * N + string; j++)
            {
                printf("0.00 ");
                m++;
            }
            i = m;
        }

        printf("%.2f ", InputMatrix[i]);

        if ((i + 1) % N == 0)
        {
            printf("\n");
            string++;
        }
    }


    printf("\ndet A = %.2f \n", det(InputMatrix, N));
    //if (recording > 0)
        printf("Time of execution =  %.2f\n", run_time);
    //else printf("Time of execution =  %.2f\n", end - start2);

    cudaFree(MatrixDeviceMemory); //Make the memory free
    return 0;
}
