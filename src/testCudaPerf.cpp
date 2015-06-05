///////////////////////
//TestCudaPerf.cpp
//~Matthew Wade
//6/4/15
//This contains a series of functions that test the performance of
//previously written CUDA kernels.
///////////////////////
//#include <cuda.h>
//#include <CudaLib.h>
//#include <ClassLib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
//Declare prototypes
int optimizeGPULib();

///////////////////////
//Main Body
int main (){
	
	//test performance
	optimizeGPULib();

return 0;
}

///////////////////////
//optimizeGPULib
//This is a general function created to work with a massive array 
//and to see how the GPU compares to the CPU when processing it.
//Various kernels that work with arrays will be included in this
//function as time goes on.  Ideally this function will make it
//possible to effectively test kernels or new data types.
int optimizeGPULib(){
	//Declare timer
	clock_t start;
	double cpuDur;
	double gpuDur;

	//Declare variables that will be used in all cases
	int rows = 100000;
	int cols = 10;

	double arr[rows][cols];
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++)
			arr[i][j] = (double)(i + j);
	}

	printf("ARRAY DONE\n");
	//Declare sum arrays and set to 0;
	double colSum[cols];
	for (int i = 0; i < cols; i++){
		colSum[i] = 0;
	}
	double rowSum[rows];
	for (int i = 0; i < rows; i++){
		rowSum[i] = 0;
	}
	//CPU//
	printf("start CPU\n");
	//Start timer
	start = clock();
	//Sum rows for each column of array (colSum)
	for(int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			printf("VALUE %d %d\n", i, j);
			rowSum[i] += arr[i][j];
		}
	}
	//Stop timer
	cpuDur = (clock() - start)/(double)CLOCKS_PER_SEC * 1000;
	
	//Display results
	for(int i = 0; i < 10; i++)
		printf("%d %f\n", i, rowSum[i]);
	
	//END CPU
	//GPU
	//Start Timer
	start = clock();
	
	//Declare pointers
	double *dev_arr, *dev_rowSum;

	//Declare sizes
	size_t arrSize = sizeof(double) * cols * rows;
	size_t sumSize = sizeof(double) * rows;	
	//Allocate memory on Host
	//gpuErr(cudaMalloc(&dev_arr, arrSize));
	//gpuErr(cudaMalloc(&dev_rowSum, sumSize));

	printf("Memory is allocated\n");

	//Copy memory over to host
	//gpuErr(cudaMemcpy(dev_arr, arr, arrSize, cudaMemcpyHostToDevice));
	//gpuErr(cudaMemcpy(dev_rowSum, &rowSum, sumSize, cudaMemcpyHostToDevice));

	printf("Memorey is done\n");

	//Define grid and block
	int block = 512;
	int grid = ceil(rows/block);

	//Run Kernel
	//SumRow<<<grid, block>>>(dev_arr, dev_rowSum, rows, cols);
	
	//Copy memory
	//gpuErr(cudaMemcpy(rowSum, dev_rowSum, sumSize, cudaMemcpyDeviceToHost));

	//Stop timer
	gpuDur = (clock() - start)/(double)CLOCKS_PER_SEC * 1000;

	//Display results
	for (int i = 0; i < 10; i++)
		printf("%d %f\n", i, rowSum[i]);

	//Print times
	printf("CPUTIME: %fms\n", cpuDur);
	printf("GPUTIME: %fms\n", gpuDur);
return 0;
}
