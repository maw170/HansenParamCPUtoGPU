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
__global__ void SumCol (double *dataIn, double *dataOut, int rows, int cols){
	
	//Define thread index
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("dataIn: %d %d\n", thread_id, dataIn[thread_id]);
	//Define variables to be used
	double tmpSum = 0;
	
	//Test to see if thread is within table limits
	if (thread_id < cols){

		//Loop through rows and sum	
		for(int i = 0; i < rows; i++){
			tmpSum += dataIn[i*cols + thread_id];
		}
		//Wait for all threads to finish
		__syncthreads(); 
		
		//Enter data into global memory
		dataOut[thread_id] = tmpSum;
	}
}

///////////////////////////////
//SumRow
///////////////////////////////
//Kernel sums the rows of a large array returning a 1xN matrix
//Kernel recieves and returns doubles
__global__ void SumRow (double *dataIn, double *dataOut, int rows, int cols){

	//Define thread index
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	
	//Define variables to be used
	double tmpSum = 0;

	//Test to see if thread is within table limits
	if (thread_id < rows){
	
		//Loop through columns and sum
		for(int i = 0; i < cols; i++){
			tmpSum += dataIn[thread_id*cols + i];
		}
		//Wait for all threads to finish
		__syncthreads(); 
	
		//Enter data into global memory
		dataOut[thread_id] = tmpSum;
	}	
}

///////////////////////////////
//FillArrayDouble
///////////////////////////////
//Kernel fills a given array with a given number
//Since array is unwrapped, this program can deal 
//with 1D or 2D arrays (possibly 3D though not tested yet)
__global__ void FillArrayDouble (double *dataOut, double *fillVal){

	//Define thread index
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	//Fill received array with fillVal
	dataOut[thread_id] = fillVal[0];
}

///////////////////////////////
//FillArrayFloat
///////////////////////////////
//Kernel fills a given array with a given number
//Since array is unwrapped, this program can deal 
//with 1D or 2D arrays (possibly 3D though not tested yet)
__global__ void FillArrayFloat (float *dataOut, float *fillVal){

	//Define thread index
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	//Fill received array with fillVal
	dataOut[thread_id] = fillVal[0];
}

///////////////////////////////
//FillArrayInt
///////////////////////////////
//Kernel fills a given array with a given number
//Since array is unwrapped, this program can deal 
//with 1D or 2D arrays (possibly 3D though not tested yet)
__global__ void FillArrayInt (int *dataOut, int *fillVal){

	//Define thread index
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	//Fill received array with fillVal
	dataOut[thread_id] = fillVal[0];
}
