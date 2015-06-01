#include <string>      //for handling file paths
#include <iostream>    //for debug/progress reports
#include <fstream>     //for accessing and changing values in .in and .data files
#include <math.h>

using namespace std;

string line;

double const no = 6.022 * pow(10, 23);
double const cm3 = pow(10, -24);

int nmolec;

double t;
double tote;
double evdwl;
double ecoul;
double elong;
double vol;

double mass = (12.011 * 440 + 1.008 * 660) * (1 / no);

int numa = 0;

int main(){

	ifstream control;
	control.open("CAL.ctrl");
	getline(control, line);
	control >> nmolec;
	getline(control, line);

	for (int z = 1; z <= 3; z++){
		getline(control, line);
	}
	control >> mass;
	mass = mass / no * nmolec;
	control.close();
	
	ifstream data;
	data.open("energy.out");
	
	while (getline(data, line))
	{
		numa ++;
	}
	numa = numa-2;
	
	double density[numa]; 
	double timeline[numa];
	double volume[numa];
	double avg = 0;
	double total = 0;

	data.close();
	data.open("energy.out");
	getline(data, line);
	getline(data, line); //jump around two comment lines
	
	for (int c = 0; c < numa; c++){
		data >> t >> tote >> evdwl >> ecoul >> elong >> vol;
		getline(data, line);
		
		vol = vol * cm3;
		density[c] = mass / vol;
		timeline[c] = t;
		volume[c] = vol;
	}
	
	data.close();
	ofstream output;
	output.open("density.val");

	for (int a = 0; a < numa; a++){
		output << timeline[a] << "\t" << density[a] << "\t" << volume[a] << "\t" << mass << "\n";
	
		total += density[a];
	}
	avg = total / numa;

	output << "Average:\t" << avg << "\n";

	output.close();

	
	return 0;
}
