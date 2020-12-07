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

//blockDim  - size of block
//blockIdx  - index of current block
//threadIdx - index of current thread in block

__device__ void elem(double* ar, int m, int n, double k, int N) //execution on Device
{
    int tid = blockIdx.y * blockDim.y + threadIdx.y;
    if (tid < N) 
        ar[m * N + tid] -= k * ar[n * N + tid];
}

__global__ void triangle_kernel(double* arr, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int j;
    double kof;
    for (j = 0; j < N * N; j++)
    {
        // if (!arr[j * N + j]) elem(arr, j, N - 1, N, N);
        if (tid >= j && tid < N - 1)
        {
            kof = arr[(tid + 1) * N + j] / arr[j * N + j];
            elem(arr, tid + 1, j, kof, N);
        }
    }
}


//Перемножаем элементы на главной диагонали уже треугольной матрицы, тем самым получаем определитель
double det(double* arr, int N)
{
    double d = 1.0;
    for (int i = 0; i < N; i++)  d *= arr[i * N + i];
    return d;
}


void print_cuda_device_info(cudaDeviceProp& prop)
{
    printf("Device name:                                        %s\n", prop.name);
    printf("Global memory available on device:                  %zu\n", prop.totalGlobalMem);
    printf("Shared memory available per block:                  %zu\n", prop.sharedMemPerBlock);
    printf("Warp size in threads:                               %d\n", prop.warpSize);
    printf("Maximum number of threads per block:                %d\n", prop.maxThreadsPerBlock);
    printf("Maximum size of each dimension of a block[0]:       %d\n", prop.maxThreadsDim[0]);
    printf("Maximum size of each dimension of a block[1]:       %d\n", prop.maxThreadsDim[1]);
    printf("Maximum size of each dimension of a block[2]:       %i\n", prop.maxThreadsDim[2]);
    printf("Maximum size of each dimension of a grid[0]:        %i\n", prop.maxGridSize[0]);
    printf("Maximum size of each dimension of a grid[1]:        %i\n", prop.maxGridSize[1]);
    printf("Maximum size of each dimension of a grid[2]:        %i\n", prop.maxGridSize[2]);
    printf("Clock frequency in kilohertz:                       %i\n", prop.clockRate);
    printf("totalConstMem:                                      %zu\n", prop.totalConstMem);
    printf("Major compute capability:                           %i\n", prop.major);
    printf("Minor compute capability:                           %i\n", prop.minor);
    printf("Number of multiprocessors on device:                %i\n", prop.multiProcessorCount);
}

__host__ int main()
{
    int N;
    printf("Input size of matrix N = ");
    scanf_s("%i", &N);
    unsigned int timer;


    int Matrix_size = N * N;//Size of matrix
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
        printf("%0.2f ", InputMatrix[i]);
        if (((i + 1) % N == 0) && (i != 0)) printf("\n");
    }
    printf("\n");
    _getch();


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    //print_cuda_device_info(prop);
    double* MatrixDeviceMemory;
    cudaMalloc((void**)&MatrixDeviceMemory, MatrixTotalMemory);//Выделяем память под массив на GPU
    cudaMemcpy(MatrixDeviceMemory, InputMatrix, MatrixTotalMemory, cudaMemcpyHostToDevice);//Копируем значения матрицы на GPU 
    dim3 gridSize = dim3(N, N, 1);//Размерность сетки блоков (dim3), выделенную для расчетов
    dim3 blockSize = dim3(1, 1, 1);//Размер блока (dim3), выделенного для расчетов


    //Инициализируем переменные для замера времени работы
    float recording;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //float start2 = clock(); //Fix the begin of work in timeline.

    triangle_kernel <<< gridSize, blockSize >>> (MatrixDeviceMemory, N); //Execution of matrix triangling
    cudaThreadSynchronize();
    cudaEventSynchronize(stop); 

    //float end = clock();   //Fix the end of execution

    //Получаем время работы
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&recording, start, stop);

    cudaMemcpy(InputMatrix, MatrixDeviceMemory, MatrixTotalMemory, cudaMemcpyDeviceToHost);//Копируем новую матрицу с GPU на CPU
    

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
        printf("Time of execution =  %.2f\n", recording);
    //else printf("Time of execution =  %.2f\n", end - start2);

    cudaFree(MatrixDeviceMemory); //Make the memory free
    return 0;
}
