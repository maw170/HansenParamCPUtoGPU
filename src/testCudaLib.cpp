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
	//declar variables
	cudaError_t oops;
	double arr[][5] = {	{1, 2, 3, 4, 5},
				{5, 4, 3, 2, 1}};
	double ret[][5] = 	{0, 0, 0, 0, 0};
	double ret2[][5] = 	{0, 0, 0, 0, 0};
	double *dev_arr, *dev_ret, *dev_ret2;
	size_t size = sizeof(arr);
	size_t size2 = sizeof(arr)/sizeof(arr[0]);
	
	cout << "size of arr " << size << "\n";
	cout << "size of arr " << size2 << "\n";
	
	//allocate memory for arrays on gpu
	oops = cudaMalloc(&dev_arr, size);
	if (oops != cudaSuccess) printf("Failed alloocate dev_arr\n");

	oops = cudaMalloc(&dev_ret, size);
	if (oops != cudaSuccess) printf("Failed alloocate dev_ret\n");

	oops = cudaMalloc(&dev_ret2, size);
	if (oops != cudaSuccess) printf("Failed alloocate dev_ret2\n");

	oops = cudaMemcpy(dev_arr, arr, size, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) printf("Failed cpy dev_arr\n");

	oops = cudaMemcpy(dev_ret, ret, size, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) printf("Failed cpy dev_ret\n");

	oops = cudaMemcpy(dev_ret2, ret2, size, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) printf("Failed cpy dev_ret2\n");

	
	//Run Cuda kernel from CudaLb.h
	SumCol<<<5,1>>>(dev_arr, dev_ret);
	oops = cudaMemcpy(&ret, dev_ret, size, cudaMemcpyDeviceToHost);
	if (oops != cudaSuccess) printf("Failed cpy ret\n");

	for(int a = 0; a < 5; a++)	
		cout << "SumCol " << ret[a] << "\n";

	SumRow<<<5,1>>>(dev_arr, dev_ret);
	oops = cudaMemcpy(&ret2, dev_ret2, size, cudaMemcpyDeviceToHost);
	if (oops != cudaSuccess) printf("Failed cpy ret_2\n");
	for(int a = 0; a < 5; a++)
		cout << "SumRow " << *ret2[a] << "\n";

	//Clean up memory to be nice
	cudaFree((void *) dev_arr);
	cudaFree((void * )dev_ret);

return(0);
}
