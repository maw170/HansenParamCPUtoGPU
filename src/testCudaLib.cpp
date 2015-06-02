//////////////////////////////
//Small program to test CudaLib.h
//////////////////////////////
//Currently tests:
// SumRow
// SumCol
// FillArray
//////////////////////////////
#include <cuda.h>
#include <CudaLib.h>
#include <iostream>
#include <driver_types.h>
#include <stdio.h>

using namespace std;
//////////////////////////////
//Declare prototypes
int testSumColSumRow (cudaError_t oops);
int testFillArray (cudaError_t oops);


//Body of program
int main(){
	//Declare general variables
	cudaError_t oops = cudaSuccess; //checks to see if Cuda code worked

	//Test SumCol and SumRow
	//testSumColSumRow(oops);

	//Test Fill Array
	testFillArray(oops);
	
return(0);
}

//Functions
///////////////////////////
//testSumColSumRow
//Tests the two summation functions for column and rows using 
//Cuda kernels.
int testSumColSumRow (cudaError_t oops){
	//Declare variables
	double arr[2][5] = {	{1, 2, 3, 4, 5},
				{5, 4, 3, 2, 1}};
	double ret[5] = 	{0, 0, 0, 0, 0};
	double ret2[5] = 	{0, 0, 0, 0, 0};
	double *dev_arr, *dev_ret, *dev_ret2;
	int cols = 5;
	int *dev_cols;
	int rows = 2;
	int *dev_rows;
	size_t size = sizeof(int);
	size_t size2 = sizeof(double) * 5;
	size_t size3 = sizeof(double) * 10;

	//Allocate memory	
	oops = cudaMalloc(&dev_arr, size3);
	if (oops != cudaSuccess) printf("Failed alloocate dev_arr\n");
	
	oops = cudaMalloc(&dev_cols, size);
	if (oops != cudaSuccess) printf("Failed to allocate dev_cols\n");

	oops = cudaMalloc(&dev_rows, size);
	if (oops != cudaSuccess) printf("Failed to allocate dev_rows\n");

	oops = cudaMalloc(&dev_ret, size2);
	if (oops != cudaSuccess) printf("Failed alloocate dev_ret\n");

	oops = cudaMalloc(&dev_ret2, size2);
	if (oops != cudaSuccess) printf("Failed alloocate dev_ret2\n");

	//Copy memory to device
	oops = cudaMemcpy(dev_arr, &arr, size3, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) printf("Failed cpy dev_arr\n");

	oops = cudaMemcpy(dev_cols, &cols, size, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) printf("Failed cpy dev_cols\n");
	
	oops = cudaMemcpy(dev_rows, &rows, size, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) printf("Failed cpy dev_rows\n");

	oops = cudaMemcpy(dev_ret, &ret, size2, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) printf("Failed cpy dev_ret\n");

	oops = cudaMemcpy(dev_ret2, &ret2, size2, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) printf("Failed cpy dev_ret2\n");
	
	//Run Cuda kernel from CudaLib.h
	SumCol<<<5,1>>>(dev_arr, dev_ret, dev_rows, dev_cols);
	oops  = cudaMemcpy(ret, dev_ret, size2, cudaMemcpyDeviceToHost);
	if (oops != cudaSuccess) printf("Failed cpy ret\n");

	for(int a = 0; a < 5; a++)	
		cout << "SumCol " << ret[a] << "\n";

	SumRow<<<2,1>>>(dev_arr, dev_ret2, dev_rows, dev_cols);
	oops = cudaMemcpy(ret2, dev_ret2, size2, cudaMemcpyDeviceToHost);
	if (oops != cudaSuccess) printf("Failed cpy ret_2\n");
	for(int a = 0; a < 5; a++)
		cout << "SumRow " << ret2[a] << "\n";

	//Clean up memory to be nice
	cudaFree((void *) dev_arr);
	cudaFree((void * )dev_ret);
	cudaFree(dev_ret2);
	cudaFree(dev_cols);
	cudaFree(dev_rows);	
return 0;
}

///////////////////////////
//testFillArray
//Tests the FillArray Cuda Kernel found in CudaLib.h
//Uses simple test case
int testFillArray (cudaError_t oops){

	//Declare variable
	int numBlock = 4;
	int numThread = 4;
	int filli = 99;
	int arri[numBlock][numThread];
	float fillf = 99.99;
	float arrf[numBlock][numThread];
	double filld = 27.72;
	double arrd[numBlock][numThread];
	//Declare pointers
	int *dev_filli, *dev_arri;
	float *dev_fillf, *dev_arrf;
	double *dev_filld, *dev_arrd;

	//Set sizes
	size_t sizei = sizeof(int);
	size_t sizef = sizeof(float);
	size_t sized = sizeof(double);
	size_t iArr = sizei * numBlock * numThread;
	size_t fArr = sizef * numBlock * numThread;
	size_t dArr = sized * numBlock * numThread;

	//Allocate memory on device
	oops = cudaMalloc(&dev_filli, sizei);
	if (oops != cudaSuccess) printf("Failed alloocate dev_filli\n");
	
	oops = cudaMalloc(&dev_arri, iArr);
	if (oops != cudaSuccess) printf("Failed alloocate dev_arri\n");

	oops = cudaMalloc(&dev_fillf, sizef);
	if (oops != cudaSuccess) printf("Failed alloocate dev_fillf\n");
	
	oops = cudaMalloc(&dev_arrf, fArr);
	if (oops != cudaSuccess) printf("Failed alloocate dev_arrf\n");

	oops = cudaMalloc(&dev_filld, sized);
	if (oops != cudaSuccess) printf("Failed alloocate dev_filld\n");

	oops = cudaMalloc(&dev_arrd, dArr);
	if (oops != cudaSuccess) printf("Failed alloocate dev_arrd\n");

	//Copy over values
	oops = cudaMemcpy(dev_filli, &filli, sizei, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) printf("Failed cpy dev_filli\n");

	oops = cudaMemcpy(dev_arri, &arri, iArr, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) printf("Failed cpy dev_arri\n");
	
	oops = cudaMemcpy(dev_fillf, &fillf, sizef, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) printf("Failed cpy dev_fillf\n");

	oops = cudaMemcpy(dev_arrf, &arrf, fArr, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) printf("Failed cpy dev_ret\n");

	oops = cudaMemcpy(dev_filld, &filld, sized, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) printf("Failed cpy dev_filld\n");
	
	oops = cudaMemcpy(dev_arrd, &arrd, dArr, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) printf("Failed cpy dev_arrd\n");

	//Run kernel
	FillArrayInt<<<numBlock, numThread>>>(dev_arri, dev_filli);
	oops  = cudaMemcpy(arri, dev_arri, iArr, cudaMemcpyDeviceToHost);
	if (oops != cudaSuccess) printf("Failed cpy arri\n");
	
	FillArrayFloat<<<numBlock, numThread>>>(dev_arrf, dev_fillf);
	oops  = cudaMemcpy(arrf, dev_arrf, fArr, cudaMemcpyDeviceToHost);
	if (oops != cudaSuccess) printf("Failed cpy arri\n");
	

	FillArrayDouble<<<numBlock, numThread>>>(dev_arrd, dev_filld);
	oops  = cudaMemcpy(arrd, dev_arrd, dArr, cudaMemcpyDeviceToHost);
	if (oops != cudaSuccess) printf("Failed cpy arri\n");
	
	//Display results	
	for(int i = 0; i < numBlock; i++){
		for(int j = 0; j < numThread; j++){
			printf("%d ", arri[i][j]);
		}	
		printf("\n");
	}

	for(int i = 0; i < numBlock; i++){
		for(int j = 0; j < numThread; j++){
			printf("%f ", arrf[i][j]);
		}	
		printf("\n");
	}			

	for(int i = 0; i < numBlock; i++){
		for(int j = 0; j < numThread; j++){
			printf("%f ", arrd[i][j]);
		}	
		printf("\n");
	}	

return 0;
}
