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
//Program Specific Kernels
/////////////////////////////////////////////
//Fill2DArray
//Fills two different arrays with a given value
//Kernel can only process doubles
__global__ void Fill2DArray (double *data1, double *data2, double *fillVal){

	//Define thread index
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	//Fill arrays with fillVal
	data1[thread_id] = fillVal[0];
	data2[thread_id] = fillVal[0];
}

//Fill1DArray
//Fills single array with a given value
//Kernel can only process doubles
__global__ void Fill1DArray (double *data1, double *fillVal){
	//Define thread index
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	
	//Fill array with fillVal
	data1[thread_id] = fillVal[0];
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

	//START CUDA CODE//
	start = clock();
	//Define pointers and sizes
	double fill = 0;
	double *dev_molecvdwl, *dev_moleccoul, *dev_molectstep, *dev_fill;
	size1 = sizeof(double) * nimage;
	size2 = sizeof(double) * nimage * nmolec;

	//Allocate memory
	oops = cudaMalloc(&dev_molecvdwl, size2);
	if (oops != cudaSuccess) cout << "Failed to allocate dev_molecvdwl\n";

	oops = cudaMalloc(&dev_moleccoul, size2);
	if (oops != cudaSuccess) cout << "Failed to allocate dev_moleccoul\n";

	oops = cudaMalloc(&dev_molectstep, size1);
	if (oops != cudaSuccess) cout << "Failed to allocate dev_molectstep\n";

	oops = cudaMalloc(&dev_fill, sizeof(double));
	if (oops != cudaSuccess) cout << "Failed to allocate dev_fill\n";

	//Copy memory to device
	oops = cudaMemcpy(dev_molecvdwl, &molecvdwl, size2, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) cout << "Failed to cpy dev_molecvdwl\n";
	
	oops = cudaMemcpy(dev_moleccoul, &moleccoul, size2, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) cout << "Failed to cpy dev_molecvdwl\n";
	
	oops = cudaMemcpy(dev_molectstep, &molectstep, size1, cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) cout << "Failed to cpy dev_molecvdwl\n";
	
	oops = cudaMemcpy(dev_fill, &fill, sizeof(double), cudaMemcpyHostToDevice);
	if (oops != cudaSuccess) cout << "Failed to cpy dev_molecvdwl\n";
	
	//Run kernel
	Fill2DArray<<<nmolec, nimage>>>(dev_molecvdwl, dev_moleccoul, dev_fill);
	Fill1DArray<<<1, nimage>>>(dev_molectstep, dev_fill);

	//Copy memory from device to host
	oops = cudaMemcpy(molecvdwl, dev_molecvdwl, size2, cudaMemcpyDeviceToHost);
	if (oops != cudaSuccess) cout << "Failed to cpy dev_molecvdwl\n";
	
	oops = cudaMemcpy(moleccoul, dev_moleccoul, size2, cudaMemcpyDeviceToHost);
	if (oops != cudaSuccess) cout << "Failed to cpy dev_molecvdwl\n";
	
	oops = cudaMemcpy(molectstep, dev_molectstep, size1, cudaMemcpyDeviceToHost);
	if (oops != cudaSuccess) cout << "Failed to cpy dev_molecvdwl\n";

	//Free memory to be nice
	cudaFree(dev_fill);
	cudaFree(dev_molecvdwl);
	cudaFree(dev_moleccoul);
	cudaFree(dev_molectstep);

	duration = (clock() - start)/(double) CLOCKS_PER_SEC;
	//END CUDA CODE//

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

	for(int cc = 0; cc < nimage; cc++){
		cvdwltmp = cvdwl[cc];
		ccoultmp = ccoul[cc];
		
		for (int bb = 0; bb < nmolec; bb++){ //cycle through all molecules to add Ei - Ec to total for all time steps
			dvdwl[bb] = dvdwl[bb] + molecvdwl[cc][bb] - cvdwltmp/nmolec;
			dcoul[bb] = dcoul[bb] + moleccoul[cc][bb] - ccoultmp/nmolec;
		}
		voltot = voltot + cvol[cc]; //calc total volume across time steps
		dumdouble = 0;
	}

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

