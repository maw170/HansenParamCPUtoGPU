//////////////////////////////
//MultiDimKernelLaunch.cpp
//This program is an example posted online concerning the
//launch of multiple processes in a 2D format.  The hope 
//is that this program also contains incormation concerning
//the passing and use of 2D arrays using CUDA.  This program
//is probably written in C.
/////////////////////////////

////////////////////////////
//Libraries
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

///////////////////////////
//Define Kernel
__global__ void kernel(int *array){
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	
	//map 2d indicies to single liner 1d index
	int grid_width = gridDim.x * blockDim.x;
	int index = index_y * grid_width + index_x; //every value of y goes to next "row"
	
	//map the 2d block indicies to single linear 1d block index
	int result = blockIdx.y * gridDim.x + blockIdx.x;
	
	//write out result
	array[index] = result;
}

int main(void){

	//define size of grid in box x and y direction
	int num_elements_x = 16;
 	int num_elements_y = 16; 
	
	//setup size of memory that we will be requesting
	int num_bytes = num_elements_x * num_elements_y * sizeof(int);
	
	int *device_array = 0;
	int *host_array = 0;

	//allocate memory to host and device
	host_array = (int*)malloc(num_bytes);
	cudaMalloc((void**)&device_array, num_bytes);
	
	//create 2d 4x4 thread blocks
	dim3 block_size;
	block_size.x = 4;
	block_size.y = 4;
	
	//configure a 2d grid
	dim3 grid_size;
	grid_size.x = num_elements_x / block_size.x;
	grid_size.y = num_elements_y / block_size.y;
	
	//gridsize and block size are passed as arugments to kernel
	kernel <<<grid_size, block_size>>> (device_array);
		//note how system defines 2d run with grid size as "x" and block_size
		//(aka number of threads/block) as "y"
	
	//download results and inspect on host
	cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);
	
	//print out result in nested loop
	for(int row = 0; row < num_elements_y; row++){
		for(int col = 0; col < num_elements_x; col++){
			printf("%2d ", host_array[row*num_elements_x + col]);
		}
		printf("\n");
	}
	printf("\n");

	//free memory to be nice
	free(host_array);
	cudaFree(device_array);

return 0;
}
