///////////////////////
//TestCudaPerf.cpp
//~Matthew Wade
//6/4/15
//This contains a series of functions that test the performance of
//previously written CUDA kernels.
///////////////////////
#include <cuda.h>
#include <CudaLib.h>
//#include <ClassLib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
using namespace std;
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

	vector< vector<double> > arr(rows, vector<double>(cols));
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++)
			arr[i][j] = (double)(i + j);
	}
	
	//Declare sum arrays and set to 0;
	vector<double> colSum(cols);
	for (int i = 0; i < cols; i++){
		colSum[i] = 0;
	}
	vector<double> rowSum(rows);
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
			rowSum[i] += arr[i][j];
		}
	}

	//Stop timer
	cpuDur = (clock() - start)/(double)CLOCKS_PER_SEC * 1000;
	
	//Display results
	for(int i = 0; i < 10; i++)
		printf("%d %f\n", i, rowSum[i]);
	
	//END CPU
	
	//Set Sum arrays to 0
	for (int i = 0; i < cols; i++){
		colSum[i] = 0;
	}
	for (int i = 0; i < rows; i++){
		rowSum[i] = 0;
	}
	
	//GPU
	//Start Timer
	start = clock();
	
	//Declare pointers
	//double *dev_arr, *dev_rowSum;

	//Declare sizes
	size_t arrSize = sizeof(double) * cols * rows;
	size_t sumSize = sizeof(double) * rows;	
	//Allocate memory on Host
	//gpuErr(cudaMalloc(&dev_arr, arrSize));
	//gpuErr(cudaMalloc(&dev_rowSum, sumSize));
	thrust::device_vector <double> dev_arr[rows][cols];
	dev_arr = arr;
	thrust::device_vector <double> dev_colSum = colSum;

	//Copy memory over to host
	//gpuErr(cudaMemcpy(dev_arr, arr, arrSize, cudaMemcpyHostToDevice));
	//gpuErr(cudaMemcpy(dev_rowSum, &rowSum, sumSize, cudaMemcpyHostToDevice));
	double* dev_arr_ptr = thrust::raw_pointer_cast(&dev_arr[0][0]);
	double* dev_colSum_ptr = thrust::raw_pointer_cast(&dev_colSum[0]);

	//Define grid and block
	int block = 512;
	int grid = ceil(rows/block);

	//Run Kernel
	SumRow<<<grid, block>>>(dev_arr_ptr, dev_colSum_ptr, rows, cols);
	
	//Copy memory
	//gpuErr(cudaMemcpy(rowSum, dev_rowSum, sumSize, cudaMemcpyDeviceToHost));
	thrust::host_vector <double> tmp[cols];
	tmp = dev_colSum;
	colSum = tmp;
	
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
