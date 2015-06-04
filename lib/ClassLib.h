/////////////////////////////
//ClassCudaLib.h
//~Matthew Wade
//6/2/15
/////////////////////////////
//General library containing classes and structs associated
//with Cuda processing.  These classes will be designed in a 
//very general manner to ensure that they can be used in a variety
//of situations.
/////////////////////////////
#include <cuda.h>

//Include guards
#ifndef CLASSCUDALIB_H
#define CLASSCUDALIB_H


/////////////////////////////
//gpuAssert (aka gpuErr)
/////////////////////////////
//Function serves as an error checker when running GPU code.
//Prints out error, file, and line where it occured.
//Code was taken from "Jason R. Mick"'s anwer on stackoverflow.
//url: stackoverflow.com/questions/9676441/cudas-cudamemcpytosymbol-throwsinvalid-argument-error
/////////////////////////////
__inline__ __host__ void gpuAssert(cudaError_t code, char *file, int line){
	//set bool to decide whether to abort or not
	bool abort = false;
		
	//print error to stderr
	if (code != cudaSuccess){
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line;
	}	
	
	//if already decided to abort then exit code
	if (abort) exit(code);
}
//Define function that will actually call this function
#define gpuErr(ans) {gpuAssert((ans), __FILE__, __LINE__);}

/////////////////////////////
//ArrayProp
/////////////////////////////
//Class consists of a generic array that also holds information
//related to the dimensions of the array.  This class serves a 
//very similar purpose to vectors, except that it is designed to
//be used in cuda code and premptivly sets up pitch and
//This object will be passed to a Cuda Kernel where 
//it should still have access to all components.
/////////////////////////////
typedef struct ArrayProp
{
	//define variables
	int rows;
	int cols;
	double *arr; //Defined as pointer since size is not known
} ArrayProp;

/////////////////////////////
//OptGridBlock
/////////////////////////////
//Optimize the block and grid size based on the size of an array
//The program will hopefully prevent the block and grid size from
//being defined as magic numbers.  The program is written under the
//assumption that the second gpu will not be used.
//Program is designed to setup 2D array of processors and threads
//Program also assumes that 
/////////////////////////////
int OptGridBlock (int rows, int cols){
	//Define general variables
	int totProc = 14; //This is the total number of processors on a single GPU.
	int totWarp = 512; //This is the total number of warps on a single GPU
	int perWarp = 32;
	dim3 grid, block;

	//Calculate the max number of warps (32 threads)
	int maxWarp = ceil((rows*cols)/perWarp * 1.0);
	
	

	
}

/////////////////////////////
//GenTestArray
//Generates massive table using nested for loops.
//Returns the massive table and nothing else.  It is
//assumed that the test function calculates totals. This 
//function is designed a test case when programming a 
//CUDA Kernel.
//NOTE: This function is not written to be fast!
//	It is written to give a simple result!
/////////////////////////////
int GenTestArray(int rows, int cols)
{
	int arr[rows][cols];

	for (int j = 0; j < cols; j++){
		for (int i = 0; i < rows; i++){
			arr[i][j] = i + j;
		}
	}
	return arr;
}

//END INCLUDE GUARDS
#endif
