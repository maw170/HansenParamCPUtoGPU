//////////////////////////////////////////////
//Initial setupa and libraries to be included
#include <string>      //for handling file paths
#include <iostream>    //for debug/progress reports
#include <fstream>     //for accessing and changing values in .in and .data files
#include <math.h>
#include <sstream>
#include <unistd.h>
#include <stdlib.h>

using namespace std;

//////////////////////////////////////////////
//Declar variables to be used in program   //
////////////////////////////////////////////
string job = "";
int njobs = 0;

int nimage = 80;		//number of images in intmole.dat
int nmolec = 110;	//number of molecules TO BE SET BY USER
int nline = nimage;		//number of lines in energy.out file

double evdwlfromfile = 0;
double ecoulfromfile = 0;
double elongfromfile = 0;
double etailfromfile = 0;

double cvdwltmp = 0;
double ccoultmp = 0;
double vdwltot = 0;  //from here down, must reset each time through outer loop
double coultot = 0;
double voltot = 0;
double volavg = 0;


double hc = 0;
double hv = 0;
double hc2 = 0;
double hv2 = 0;
double hildebrand = 0;

bool nanflag = false;

stringstream command;
stringstream directory;
stringstream filename;
int dumint = 0;
double dumdouble = 0;
string line = "";
string dumstr = "";



/////////////////////////////////////////////
//Declare constants to be used in program //
///////////////////////////////////////////
double const AVGNUM = 6.022 * pow(10.0, 23.0);

/////////////////////////////////////////////
//Declare prototypes			  //
///////////////////////////////////////////

////////////////////////////////////////////
//Declare important objects		 //
//////////////////////////////////////////
ifstream cell;
ifstream mole;
ifstream control;

////////////////////////////////////////////
//Begin main program                     //
//////////////////////////////////////////
int main(int argc, char *argv[]){

	job = 1;
	njobs = 1;

	//Define Timer
	clock_t start;
	double duration;

	//Read data from universal data file
	control.open("CAL.ctrl");
	if(!control.is_open()){
		cout << "NO CONTROL: MOVE CAL.ctrl TO SAME DIRECTORY";
		abort();
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

	//now in directory with data
	cell.open("energy.out"); //open the two input files to be used
	mole.open("intmolec.dat"); //open the two input files to be used

	filename.str("");	

	if(mole.is_open()){
		cout << "MOLEC DATA OPEN\n";
	}
	else{
		cout << "ERROR: COULD NOT OPEN FILE: intmolec.out\n";
		abort();
	}
	if(cell.is_open()){
		cout << "CELL DATA OPEN\n";
	}
	else{
		cout << "ERROR: COULD NOT OPEN FILE: energy.out\n";
		abort();
	}
	double molecvdwl[nimage][nmolec]; //= {0}; // !!setup 2d array 
	double moleccoul[nimage][nmolec]; //= {0}; // !!setup 2d array 
	double molectstep[nimage]; //= {0};
	
	start = clock();
	for(int r = 0; r < nimage; r++){
		molectstep[r] = 0;
		for (int rr = 0; rr < nmolec; rr++){
			molecvdwl[r][rr] = 0;
			moleccoul[r][rr] = 0;
		}
	}
	duration = (clock()-start)/(double) CLOCKS_PER_SEC;

	//emole -> retreive energy for each time step for each molecule
	for(int c = 0; c < nimage; c++){
		getline(mole, line);
		mole >> molectstep[c];
		getline(mole, line);

		for(int b = 0; b < nmolec; b++){
			//get all molecular data for image
			mole >> dumint >> molecvdwl[c][b] >> moleccoul[c][b];
			getline(mole, line); //snag new line and move to next
			//cout << c << "\t" << b << "\t" << molectstep[c] << "\t" << molecvdwl[c][b] << "\t" << moleccoul[c][b] << "\n";

		}
	}
	//ABOVE METHOD MIGHT HAVE MEMORY ISSUES, PLEASE FIND DIFFERENT WAY
	//ecell -> retrieve two types of energy and volume at time step t

	double tstep[nline]; // !!setup arrays b/c c++ cant allocate 
	double cvdwl[nline]; 
	double ccoul[nline];
	double cvol[nline];
	for(int r = 0; r < nline; r++){ //setup all values to zero
		tstep[r] = 0;
		cvdwl[r] = 0;
		ccoul[r] = 0;
		cvol[r] = 0;
	}

	getline(cell, line);
	getline(cell, line);

	for(int a = 0; a < nline; a++){
		cell >> tstep[a] >> dumdouble >> evdwlfromfile >> etailfromfile >> ecoulfromfile >> elongfromfile >> cvol[a];
		getline(cell, line); //snag new line and move to next

		cvdwl[a] = evdwlfromfile + etailfromfile;
		ccoul[a] = ecoulfromfile + elongfromfile; //total coulomb energy is sum of sort and long range energies
		etailfromfile = 0;
		evdwlfromfile = 0;
		ecoulfromfile = 0;
		elongfromfile = 0;

		cout << tstep[a] << "\t" << cvdwl[a] << "\t" << ccoul[a] << "\t" << cvol[a] << "\n";
	}

	//Check to see time steps match
	if(nline != nimage){
		cout << "The number of images from cell is: " << nline << "\n";
		cout << "The number of images from molecule is: " << nimage << "\n";
		abort(); 
	}

	double dvdwl[nmolec];
	double dcoul[nmolec];
	for(int r = 0; r < nmolec; r++){ //set all values in array to zero
		dvdwl[r] = 0;
		dcoul[r] = 0;
	}


	//Begin to calculate Hansen solubility parameters
	for(int cc = 0; cc < nimage; cc++){
		cvdwltmp = cvdwl[cc];
		ccoultmp = ccoul[cc];
		//cout << "IMAGE:\t" << cc <<"\t"<<cvdwltmp << "\t" << ccoultmp << "\n";

		for (int bb = 0; bb < nmolec; bb++){ //cycle through all molecules to add Ei - Ec to total for all time steps
			dvdwl[bb] = dvdwl[bb] + molecvdwl[cc][bb] - cvdwltmp/nmolec;
			dcoul[bb] = dcoul[bb] + moleccoul[cc][bb] - ccoultmp/nmolec;
			//cout << "VDWL:\t" << cvdwltmp/nmolec << "\t" << dvdwl[bb] << "\t" << molecvdwl[cc][bb] << "\n";
			//cout << "COUL:\t" << ccoultmp/nmolec << "\t" << dcoul[bb] << "\t" << moleccoul[cc][bb] << "\n";
			//cout << "LOC:\t" << cc << "\t" << bb << "\n";
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

	//write results to file in main directory
	/*
	filename << "results." << job;
	dumstr = filename.str();
	const char* foochar3 = dumstr.c_str();

	ofstream output;
	output.open(foochar3, std::ios::app);

	filename.str("");

	output << "JOB\n" << directory.str() << "\n";
	output << "UNI\n";
	output << "Hildebrand\tVdwl\tCoul\n";
	output << hildebrand << "\t" << hv << "\t" << hc << "\n";
	*/
	//reset and delete all variables for next job
	nimage = 0;
	nline = 0;
	vdwltot = 0;
	coultot = 0;
	voltot = 0;
	nanflag = false;
	mole.close();
	cell.close();
	//output.close();

	//delete [] molecvdwl;
	
	cout << "TIMER: " << duration << "\n";

	return 0;
}

