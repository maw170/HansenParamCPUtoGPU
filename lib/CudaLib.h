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

///////////////////////////////
//SumCol
///////////////////////////////
//Kernel sums the columns of a large array returning a Nx1 matrix
//Kernel recieves and returns doubles
__global__ void SumCol (double *dataIn, double *dataOut){

	//Define thread index
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	
	//Define variables to be used
	double tmpSum = 0;
	int numRow = sizeof(dataIn)/sizeof(dataIn[0]);
	
	for(int i = 0; i < numRow; i++){
		tmpSum += dataIn[thread_id,i];
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
__global__ void SumRow (double *dataIn, double *dataOut){

	//Define thread index
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	
	//Define variables to be used
	double tmpSum = 0;
	int numCol = sizeof(dataIn)/sizeof(double);
	
	for(int i = 0; i < numCol; i++){
		tmpSum += dataIn[i,thread_id];
	}
	//Wait for all threads to finish
	__syncthreads(); 
	
	//Enter data into global memory
	dataOut[thread_id] = tmpSum;
}

