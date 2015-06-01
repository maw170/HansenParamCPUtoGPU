//////////////////////////////
//Small program to test CudaLib.h
//////////////////////////////
#include <cuda.h>
#include <CudaLib.h>
#include <iostream>

using namespace std;

//Body of program
int main(){
	//declar variables
	double arr[][5] = {	{1, 2, 3, 4, 5},
				{5, 4, 3, 2, 1}};
	double ret[][5] = {	{0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0}};
	int *dev_arr, *dev_ret;
	int size = sizeof(arr);
	int size2 = sizeof(arr)/sizeof(arr[0]);
	
	cout << "size of arr" << size << "\n";
	cout << "size of arr" << size2 << "\n";
	
	//allocate memory for arrays on gpu
	cudaMalloc((void**)&dev_arr, size);
	cudaMalloc((void**)&dev_ret, size);
	cudaMemcpy(dev_arr, &arr, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ret, &ret, cudaMemcpyHostToDevice);

	SumCol<<<5,1>>>(dev_arr, dev_ret);
	cout << "SumCol" << ret << "\n";

	SumRow<<<5,1>>>(dev_arr, dev_ret);
	cout << "SumRow" << ret << 

	


return(0);
}
