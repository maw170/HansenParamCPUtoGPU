///////////////////////////////
//CudaLib.h
//~Matthew Wade
//5/29/15
///////////////////////////////
//General library containing CUDA kernels that can be used in multiple
//situations.  The goal of this library is to have a number of 
//kernels that can be either modified or reused.
///////////////////////////////
#include <cuda.h>
#include <stdio.h>

///////////////////////////////
//SumCol
///////////////////////////////
//Kernel sums the columns of a large array returning a Nx1 matrix
//Kernel recieves and returns doubles
__global__ void SumCol (double *dataIn, double *dataOut, int *rows, int *cols){
	
	//Define thread index
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("dataIn: %d %f\n", thread_id, dataIn[thread_id]);
	//Define variables to be used
	double tmpSum = 0;

	//Loop through rows and sum	
	for(int i = 0; i < rows[0]; i++){
		tmpSum += dataIn[i*cols[0] + thread_id];
	}
	//Wait for all threads to finish
	__syncthreads(); 
	
	//Enter data into global memory
	dataOut[thread_id] = tmpSum;
}

///////////////////////////////
//SumRow
///////////////////////////////
//Kernel sums the rows of a large array returning a 1xN matrix
//Kernel recieves and returns doubles
__global__ void SumRow (double *dataIn, double *dataOut, int *rows, int *cols){

	//Define thread index
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	
	//Define variables to be used
	double tmpSum = 0;

	printf("WIDTH: %d \n", cols[0]);	
	//Loop through columns and sum
	for(int i = 0; i < cols[0]; i++){
		tmpSum += dataIn[thread_id*cols[0] + i];
	}
	//Wait for all threads to finish
	__syncthreads(); 
	
	//Enter data into global memory
	dataOut[thread_id] = tmpSum;
}

