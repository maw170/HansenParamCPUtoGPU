//////////////////////////////
//Small program to test CudaLib.h
//////////////////////////////
#include <cuda.h>
#include <CudaLib.h>
#include <iostream>
#include <driver_types.h>
#include <stdio.h>

using namespace std;

//Body of program
int main(){
	//declare variables
	cudaError_t oops; //checks to see if Cuda code worked
	double arr[2][5] = {	{1, 2, 3, 4, 5},
				{5, 4, 3, 2, 1}};
	double ret[5] = 	{0, 0, 0, 0, 0};
	double ret2[5] = 	{0, 0, 0, 0, 0};
	double *dev_arr, *dev_ret, *dev_ret2;
	size_t size = sizeof(double) * 10;
	size_t size2 = sizeof(double) * 5;
	
	//declare variable to sending 2D array to device
	size_t pitch;
	size_t w = 5;
	size_t h = 2;	
	//allocate memory for arrays on gpu
	oops = cudaMallocArray(&dev_arr, &pitch, w, h);
	if (oops != cudaSuccess) printf("Failed alloocate dev_arr\n");

	oops = cudaMalloc(&dev_ret, size2);
	if (oops != cudaSuccess) printf("Failed alloocate dev_ret\n");

	oops = cudaMalloc(&dev_ret2, size2);
	if (oops != cudaSuccess) printf("Failed alloocate dev_ret2\n");

	oops = cudaMemcpy2D(dev_arr, pitch, arr, pitch, w, h, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) printf("Failed cpy dev_arr\n");

	oops = cudaMemcpy(dev_ret, ret, size2, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) printf("Failed cpy dev_ret\n");

	oops = cudaMemcpy(dev_ret2, ret2, size2, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) printf("Failed cpy dev_ret2\n");

	
	//Run Cuda kernel from CudaLib.h
	SumCol<<<5,1>>>(dev_arr, dev_ret);
	oops = cudaMemcpy(ret, dev_ret, size2, cudaMemcpyDeviceToHost);
	if (oops != cudaSuccess) printf("Failed cpy ret\n");

	for(int a = 0; a < 5; a++)	
		cout << "SumCol " << ret[a] << "\n";

	SumRow<<<5,1>>>(dev_arr, dev_ret2);
	oops = cudaMemcpy(ret2, dev_ret2, size2, cudaMemcpyDeviceToHost);
	if (oops != cudaSuccess) printf("Failed cpy ret_2\n");
	for(int a = 0; a < 5; a++)
		cout << "SumRow " << ret2[a] << "\n";

	//Clean up memory to be nice
	cudaFree((void *) dev_arr);
	cudaFree((void * )dev_ret);

return(0);
}
