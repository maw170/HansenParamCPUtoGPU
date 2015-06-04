//////////////////////////////////////////////
//Initial setup and libraries to be included
#include <cuda.h>	//CUDA general library
#include <string>      	//for handling file paths
#include <iostream>    	//for debug/progress reports
#include <fstream>     	//for accessing and changing values in .in and .data files
#include <sstream>
#include <unistd.h>

using namespace std;

//////////////////////////////////////////////
//Declar variables to be used in program   //
////////////////////////////////////////////
int nimage = 0;		//number of images in intmole.dat
int nmolec = 0;		//number of molecules TO BE SET BY USER
int nline = 0;		//number of lines in energy.out file

//variables that will be read from file
double evdwlfromfile = 0;
double ecoulfromfile = 0;
double elongfromfile = 0;
double etailfromfile = 0;

//temporary variables to be used in calculation
double cvdwltmp = 0;
double ccoultmp = 0;

//total variables for coulomb and vdwl energies
double vdwltot = 0;  //from here down, must reset each time through outer loop
double coultot = 0;
double voltot = 0;
double volavg = 0;

//calculated Hildebrand and Hansen solubility parameters
double hc = 0;
double hv = 0;
double hc2 = 0;
double hv2 = 0;
double hildebrand = 0;

//Misc variables
bool nanflag = false;
int dumint = 0;
double dumdouble = 0;
string line = "";
string dumstr = "";



/////////////////////////////////////////////
//Declare constants to be used in program //
///////////////////////////////////////////
double const AVGNUM = 6.022 * pow(10.0, 23.0);

////////////////////////////////////////////
//Setup file streams			 //
//////////////////////////////////////////
ifstream fromSim;
ifstream fromPost;
ifstream control;
ofstream output;

/////////////////////////////////////////////
//Define program specific cuda kernel
//This kernel is copied from CudaLib.h with
//a few minor adjustments.  The code now
//processes two arrays instead of one while
//also subtracting an additional value
__global__ void SumCol (double *c_in, double *v_in, double *c_out, double *v_out,  double *c_tot, double *v_tot, int rows, int cols){
/*	//Setup shared variable for threads of same block to access
	extern __shared__ int vdata[];
	extern __shared__ int cdata[];
*/
	//Calculate thread ID
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int gidx = blockIdx.x * blockDim.x + threadIdx.x;
	int gidy = blockIdx.y * blockDim.y + threadIdx.y;

	//Check to see that tid does not exceed matrix dimensions
	if (gidx  < cols){
		double tmpCoul = 0;
		double tmpVdwl = 0;
		for(int i = 0; i < rows; i++){
			tmpCoul += c_in[i*cols + gidx] - c_tot[i];
			tmpVdwl += v_in[i*cols + gidx] - v_tot[i];
		}
		__syncthreads();
		c_out[gidx] = tmpCoul;
		v_out[gidx] = tmpVdwl;
		
		printf("value %d %f %f\n", gidx, c_out[gidx], v_out[gidx]);
		__syncthreads();
	}
	

} 

////////////////////////////////////////////
//Begin main program                     //
//////////////////////////////////////////
int main(int argc, char *argv[]){
	//Define generic variables
	cudaError_t oops = cudaSuccess;
	size_t size1 = 0;
	size_t size2 = 0;
	size_t size3 = 0;

	//Define timer
	clock_t start;
	double duration;

	//Read data from universal data file
	control.open("CAL.ctrl");

	//Check for error
	if(!control.is_open()){
		cout << "NO CONTROL: MOVE CAL.ctrl TO SAME DIRECTORY";
		abort(); //Throw exception
	}
	getline(control, line);
	control >> nmolec;
	getline(control, line);
	for(int z = 1; z <= 11; z++){
		getline(control, line); //jump over junk lines
	}
	control >> nimage;
	control.close();

	nline = nimage;

	//Open data from simulation
	fromSim.open("energy.out");
	if(fromSim.is_open()){
		cout << "CELL DATA OPEN\n";
	}
	else{
		cout << "ERROR: COULD NOT OPEN FILE: energy.out\n";
		abort();
	}

	//Open data from postprocessing script
	fromPost.open("intmolec.dat"); //open the two input files to be used	
	if(fromPost.is_open()){
		cout << "MOLEC DATA OPEN\n";
	}
	else{
		cout << "ERROR: COULD NOT OPEN FILE: intmolec.out\n";
		abort();
	}

	//Setup arrays to store data from postprocessing script
	double molecvdwl[nimage][nmolec]; // !!setup 2d array 
	double moleccoul[nimage][nmolec]; // !!setup 2d array 
	double molectstep[nimage];

	for(int r = 0; r < nimage; r++){
		molectstep[r] = 0;
		for (int rr = 0; rr < nmolec; rr++){
			molecvdwl[r][rr] = 0;
			moleccoul[r][rr] = 0;
		}
	}

	//Fill arrays with data from post processing
	for(int c = 0; c < nimage; c++){
		getline(fromPost, line);
		fromPost >> molectstep[c];
		getline(fromPost, line);

		for(int b = 0; b < nmolec; b++){
			//get all molecular data for image
			fromPost >> dumint >> molecvdwl[c][b] >> moleccoul[c][b];
			getline(fromPost, line); //snag new line and move to next 
		}
	}	

	//Setup arrays for parsing data from simulation
	double tstep[nimage];
	double cvdwl[nimage]; 
	double ccoul[nimage];
	double cvol[nimage];
	for(int r = 0; r < nimage; r++){ //setup all values to zero
		tstep[r] = 0;
		cvdwl[r] = 0;
		ccoul[r] = 0;
		cvol[r] = 0;
	}

	getline(fromSim, line);
	getline(fromSim, line);

	//Read data from simulation
	for(int a = 0; a < nimage; a++){
		fromSim >> tstep[a] >> dumdouble >> evdwlfromfile >> etailfromfile >> ecoulfromfile >> elongfromfile >> cvol[a];
		getline(fromSim, line); //snag new line and move to next

		//Place data in corresponding array
		cvdwl[a] = evdwlfromfile + etailfromfile;
		ccoul[a] = ecoulfromfile + elongfromfile; //total coulomb energy is sum of sort and long range energies
		etailfromfile = 0;
		evdwlfromfile = 0;
		ecoulfromfile = 0;
		elongfromfile = 0;

		cout << tstep[a] << "\t" << cvdwl[a] << "\t" << ccoul[a] << "\t" << cvol[a] << "\n";
	}

	//Check to see if number of images match (they should but this has been issue in past)
	if(nline != nimage){
		cout << "The number of images from cell is: " << nline << "\n";
		cout << "The number of images from molecule is: " << nimage << "\n";
		abort(); 
	}

	//Setup arrays to hold difference between molecule and system energies
	double dvdwl[nmolec];
	double dcoul[nmolec];
	for(int r = 0; r < nmolec; r++){ //set all values in array to zero
		dvdwl[r] = 0;
		dcoul[r] = 0;
	}


	//Begin to calculate Hansen solubility parameters
	//Will be replacing this section of code with GPU processing
	start = clock();

	//BEGIN CUDA CODE//
	//Preprocess array to be passed to kernel
	for(int cc = 0; cc < nimage; cc++){
		cvdwl[cc] = cvdwl[cc]/nmolec;
		ccoul[cc] = ccoul[cc]/nmolec;
	}

	//Define sizes
	size1 = sizeof(double) * nimage * nmolec;
	size2 = sizeof(double) * nmolec;
	size3 = sizeof(double) * nimage;
	
	//Define pointers
	double *dev_dvdwl, *dev_dcoul, *dev_molecvdwl, *dev_moleccoul, *dev_cvdwl, *dev_ccoul;

	//Allocate pointers
	oops = cudaMalloc(&dev_dvdwl, size2);
	if (oops != cudaSuccess) cout << "Failed to allocate dev_dvdwl\n";
	oops = cudaMalloc(&dev_dcoul, size2);
	if (oops != cudaSuccess) cout << "Failed to allocate dev_dcoul\n";
	oops = cudaMalloc(&dev_molecvdwl, size1);
	if (oops != cudaSuccess) cout << "Failed to allocate dev_molecvdwl\n";
	oops = cudaMalloc(&dev_moleccoul, size1);
	if (oops != cudaSuccess) cout << "Failed to allocate dev_moleccoul\n";
	oops = cudaMalloc(&dev_cvdwl, size3);
	if (oops != cudaSuccess) cout << "Failed to allocate dev_cvdwl\n";
	oops = cudaMalloc(&dev_ccoul, size3);
	if (oops != cudaSuccess) cout << "Failed to allocate dev_ccoul\n";

	//Copy values over to device memory
	oops = cudaMemcpy(dev_dvdwl, &dvdwl, size2, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) cout << "Failed to copy dev_dvdwl\n";
	oops = cudaMemcpy(dev_dcoul, &dcoul, size2, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) cout << "Failed to copy dev_dcoul\n";
	oops = cudaMemcpy(dev_molecvdwl, &molecvdwl, size1, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) cout << "Failed to copy dev_molecvdwl\n";
	oops = cudaMemcpy(dev_moleccoul, &moleccoul, size1, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) cout << "Failed to copy dev_moleccoul\n";
	oops = cudaMemcpy(dev_cvdwl, &cvdwl, size3, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) cout << "Failed to copy dev_cvdwl\n";
	oops = cudaMemcpy(dev_ccoul, &ccoul, size3, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) cout << "Failed to copy dev_ccoul\n";

	duration = (clock()-start)/(double) CLOCKS_PER_SEC;

	//Define block size
	dim3 blockSize;
	blockSize.x = 32 * ceil((double)nmolec/32);
	blockSize.y = 1;
	cout << "BLOCK SIZE: " << blockSize.x << blockSize.y << "\n";
	cout << "SIZE2 " << size2/sizeof(double) << " NUM MOLEC " << nmolec << "\n";	
	//Define grid size
	dim3 gridSize;
	gridSize.x = 1;
	gridSize.y = 1;
	
	//Call Kernels
	SumCol<<<gridSize, blockSize>>>(dev_moleccoul, dev_molecvdwl, dev_dcoul, dev_dvdwl, dev_ccoul, dev_cvdwl, nimage, nmolec);
	oops = cudaThreadSynchronize();
	if (oops != cudaSuccess) cout << "Failed to synchronize threads " << cudaGetErrorString(oops) << "\n";
	//Extract Data from kernel
	oops = cudaMemcpy(dcoul, dev_dcoul, size2, cudaMemcpyDeviceToHost);
	if (oops != cudaSuccess) cout << "Failed to copy dcoul " << cudaGetErrorString(oops) << "\n";
	oops = cudaMemcpy(dvdwl, dev_dvdwl, size2, cudaMemcpyDeviceToHost);
	if (oops != cudaSuccess) cout << "Failed to copy dvdwl " << cudaGetErrorString(oops) << "\n";

	//Calculate average volume
	for(int cc = 0; cc < nimage; cc++){
		voltot = voltot + cvol[cc]; //calc total volume across time steps
	}	

	//END CUDA CODE//

	//calculate time average of differences and average volume
	vdwltot = 0;
	coultot = 0;
	for(int aa = 0; aa < nmolec; aa++){
		dvdwl[aa] = dvdwl[aa]/nimage; //calc time average of van der waals energies 
		dcoul[aa] = dcoul[aa]/nimage; //calc time average of coulomb energies

		//calc sum for all molecules
		vdwltot = vdwltot + dvdwl[aa]; //perform summation on average van der waals energies
		coultot = coultot + dcoul[aa]; //perform summation on average coulomb energies
	}
	cout << "ENERGY TOTAL VDWL:\t" << vdwltot << "\n"; 
	cout << "ENERGY TOTAL COUL:\t" << coultot << "\n";

	volavg = voltot / nimage; //calc average volume

	volavg = volavg * pow(10, -24); //convert units from angstrom to ccm
	vdwltot = vdwltot * 1000; //convert units from kcal to cal
	coultot = coultot * 1000; //convert units from kcal to cal
	hv2 = vdwltot/(AVGNUM * volavg);
	hc2 = coultot/(AVGNUM * volavg);

	cout << "VOLUME TIMES AVGNUM: " << AVGNUM*volavg << "\n";
	cout << "VDWLTOT: " << vdwltot << "\n";
	cout << "COULTOT: " << coultot << "\n";
	cout << "HV2: " << hv2 << "\n";
	cout << "HC2: " << hc2 << "\n";

	if (hc2 < 0){
		hc2 = hc2 * (-1);
		cout << "Corrected for negative\n";
		nanflag = true;
	}

	hv = pow(hv2, .5);
	hc = pow(hc2, .5);

	cout << "HV: " << hv << "\n";
	cout << "HC: " << hc << "\n";
	//hv = hv2;
	//hc = hc2;

	if (nanflag){
		hc = hc * (-1);
	}

	hildebrand = pow(pow(hc,2.0) + pow(hv,2.0), .5);

	/*
	//write results to file		
	output.open(foochar3, std::ios::app);

	filename.str("");

	output << "JOB\n" << directory.str() << "\n";
	output << "UNI\n";
	output << "Hildebrand\tVdwl\tCoul\n";
	output << hildebrand << "\t" << hv << "\t" << hc << "\n";
	 */
	//reset and delete variables
	nimage = 0;
	nline = 0;
	vdwltot = 0;
	coultot = 0;
	voltot = 0;
	nanflag = false;
	fromSim.close();
	fromPost.close();
	output.close();


	cout << "TIMER: " << duration << "\n";

	return 0;
}

