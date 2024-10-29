#define _USE_MATH_DEFINES

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <chrono>
#include "CMatrix.h"

using namespace std;

//Note that this is a very simple serial implementation with a fixed grid and Neumann boundaries at the edges
//I am also using a vector of vectors, which is less efficient than allocating contiguous data.
CMatrix C1, C1_old, C2, C2_old;

int p, id;
int imax = 301, jmax = 301; 
double t_max = 30.0;
double t, t_out = 0.0, dt_out = 0.1, dt;
double y_max = 30.0, x_max = 30.0, dx, dy;

int rows, cols; // stores the number of rows and columns in the grid decomposition
int id_row, id_col; // stores the row and column of the processor in the grid decomposition
int imax_local, jmax_local; // variables for the local grid sizes
int i_start_local, j_start_local; // variables for the local grid start indices
vector<int> neighbors_id; // stores the ids of the neighboring processors

//set up simulation constants
const double f = 2.0, q = 0.002, epsilon = 0.03, D1 = 1.0, D2 = 0.6;

void grid_to_file(int out)
{
	//Write the output for a single time step to file
	stringstream fname1, fname2;
	fstream f1, f2;

	//File name contains the processor coordinates for easy post-processing
	fname1 << "./out/" << id_row << "_" << id_col << "_output_C1_" << out << ".dat";
	f1.open(fname1.str().c_str(), ios_base::out);
	fname2 << "./out/" << id_row << "_" << id_col << "_output_C2_" << out << ".dat";
	f2.open(fname2.str().c_str(), ios_base::out);

	//For loop excludes the padding regions
	for (int i = 1; i < imax_local - 1; i++)
	{
		for (int j = 1; j < jmax_local - 1; j++)
		{
			f1 << C1.data_2D[i][j] << "\t";
			f2 << C2.data_2D[i][j] << "\t";
		}
		f1 << endl;
		f2 << endl;
	}
	f1.close();
	f2.close();
}

void do_iteration_neumann(void)
{
	// Swap the two arrays, note - need to swap both data_1D and data_2D
	C1.swap_data(C1_old);
	C2.swap_data(C2_old);

	MPI_Request* request = new MPI_Request[4 * neighbors_id.size()];
	int request_cnt = 0;

	int imin_iter = 1;
	int imax_iter = imax_local - 1;
	int jmin_iter = 1;
	int jmax_iter = jmax_local - 1;

	// Below chunk checks if processor is on the edge of the domain and adjusts the iteration range, 
	// otherwise sends and receives the boundary values for continuous calcualtion in the middle of the domain

	// Top edge
	if (id_row != 0) {
		MPI_Isend(C1_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::up_out], neighbors_id[0], 1, MPI_COMM_WORLD, &request[request_cnt]);
		request_cnt++;
		MPI_Irecv(C1_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::up_in], neighbors_id[0], 2, MPI_COMM_WORLD, &request[request_cnt]);
		request_cnt++;

		MPI_Isend(C2_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::up_out], neighbors_id[0], 5, MPI_COMM_WORLD, &request[request_cnt]);
		request_cnt++;
		MPI_Irecv(C2_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::up_in], neighbors_id[0], 6, MPI_COMM_WORLD, &request[request_cnt]);
		request_cnt++;
	}
	else {
		imin_iter = 2;
	}

	// Bottom edge
	if (id_row != rows - 1) {
		MPI_Isend(C1_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::down_out], neighbors_id[3], 2, MPI_COMM_WORLD, &request[request_cnt]);
		request_cnt++;
		MPI_Irecv(C1_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::down_in], neighbors_id[3], 1, MPI_COMM_WORLD, &request[request_cnt]);
		request_cnt++;

		MPI_Isend(C2_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::down_out], neighbors_id[3], 6, MPI_COMM_WORLD, &request[request_cnt]);
		request_cnt++;
		MPI_Irecv(C2_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::down_in], neighbors_id[3], 5, MPI_COMM_WORLD, &request[request_cnt]);
		request_cnt++;
	}
	else {
		imax_iter = imax_local - 2;
	}

	// Left edge
	if (id_col != 0) {
		MPI_Isend(C1_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::left_out], neighbors_id[1], 4, MPI_COMM_WORLD, &request[request_cnt]);
		request_cnt++;
		MPI_Irecv(C1_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::left_in], neighbors_id[1], 3, MPI_COMM_WORLD, &request[request_cnt]);
		request_cnt++;

		MPI_Isend(C2_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::left_out], neighbors_id[1], 7, MPI_COMM_WORLD, &request[request_cnt]);
		request_cnt++;
		MPI_Irecv(C2_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::left_in], neighbors_id[1], 8, MPI_COMM_WORLD, &request[request_cnt]);
		request_cnt++;
	}
	else {
		jmin_iter = 2;
	}

	// Right edge
	if (id_col != cols - 1) {
		MPI_Isend(C1_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::right_out], neighbors_id[2], 3, MPI_COMM_WORLD, &request[request_cnt]);
		request_cnt++;
		MPI_Irecv(C1_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::right_in], neighbors_id[2], 4, MPI_COMM_WORLD, &request[request_cnt]);
		request_cnt++;

		MPI_Isend(C2_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::right_out], neighbors_id[2], 8, MPI_COMM_WORLD, &request[request_cnt]);
		request_cnt++;
		MPI_Irecv(C2_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::right_in], neighbors_id[2], 7, MPI_COMM_WORLD, &request[request_cnt]);
		request_cnt++;
	}
	else {
		jmax_iter = jmax_local - 2;
	}

	MPI_Waitall(request_cnt, request, MPI_STATUS_IGNORE);

	//Calculate the new concentrations for all the points not on the boundary of the domain
	for (int i = imin_iter; i < imax_iter; i++)
		for (int j = jmin_iter; j < jmax_iter; j++)
		{
			C1.data_2D[i][j] = C1_old.data_2D[i][j] + dt * ((C1_old.data_2D[i][j] * (1.0 - C1_old.data_2D[i][j]) - f * C2_old.data_2D[i][j] * (C1_old.data_2D[i][j] - q) / (C1_old.data_2D[i][j] + q)) / epsilon
				+ D1 * ((C1_old.data_2D[i + 1][j] + C1_old.data_2D[i - 1][j] - 2.0 * C1_old.data_2D[i][j]) / (dx * dx) + (C1_old.data_2D[i][j + 1] + C1_old.data_2D[i][j - 1] - 2.0 * C1_old.data_2D[i][j]) / (dy * dy)));

			C2.data_2D[i][j] = C2_old.data_2D[i][j] + dt * (C1_old.data_2D[i][j] - C2_old.data_2D[i][j]
				+ D2 * ((C2_old.data_2D[i + 1][j] + C2_old.data_2D[i - 1][j] - 2.0 * C2_old.data_2D[i][j]) / (dx * dx) + (C2_old.data_2D[i][j + 1] + C2_old.data_2D[i][j - 1] - 2.0 * C2_old.data_2D[i][j]) / (dy * dy)));
		}

	//Neumann boundary conditions, implement only if processor is on the edge of the domain
	if (id_row == 0) {
		for (int j = 1; j < jmax_local - 1; j++)
		{
			C1.data_2D[1][j] = C1.data_2D[2][j];
			C2.data_2D[1][j] = C2.data_2D[2][j];
		}
	}

	if (id_row == rows - 1) {
		for (int j = 1; j < jmax_local - 1; j++)
		{
			C1.data_2D[imax_local - 2][j] = C1.data_2D[imax_local - 3][j];
			C2.data_2D[imax_local - 2][j] = C2.data_2D[imax_local - 3][j];
		}
	}

	if (id_col == 0) {
		for (int i = 1; i < imax_local - 1; i++)
		{
			C1.data_2D[i][1] = C1.data_2D[i][2];
			C2.data_2D[i][1] = C2.data_2D[i][2];
		}
	}

	if (id_col == cols - 1) {
		for (int i = 1; i < imax_local - 1; i++)
		{
			C1.data_2D[i][jmax_local - 2] = C1.data_2D[i][jmax_local - 3];
			C2.data_2D[i][jmax_local - 2] = C2.data_2D[i][jmax_local - 3];
		}
	}

	t += dt;
}

void do_iteration_periodic(void)
{
	// Swap the two arrays, note - need to swap both data_1D and data_2D
	C1.swap_data(C1_old);
	C2.swap_data(C2_old);

	MPI_Request* request = new MPI_Request[4 * neighbors_id.size()];
	int request_cnt = 0;

	// Periodic communication to get and send the boundary values
	MPI_Isend(C1_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::up_out], neighbors_id[0], 1, MPI_COMM_WORLD, &request[request_cnt]);
	request_cnt++;
	MPI_Irecv(C1_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::up_in], neighbors_id[0], 2, MPI_COMM_WORLD, &request[request_cnt]);
	request_cnt++;

	MPI_Isend(C1_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::down_out], neighbors_id[3], 2, MPI_COMM_WORLD, &request[request_cnt]);
	request_cnt++;
	MPI_Irecv(C1_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::down_in], neighbors_id[3], 1, MPI_COMM_WORLD, &request[request_cnt]);
	request_cnt++;

	MPI_Isend(C1_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::left_out], neighbors_id[1], 4, MPI_COMM_WORLD, &request[request_cnt]);
	request_cnt++;
	MPI_Irecv(C1_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::left_in], neighbors_id[1], 3, MPI_COMM_WORLD, &request[request_cnt]);
	request_cnt++;

	MPI_Isend(C1_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::right_out], neighbors_id[2], 3, MPI_COMM_WORLD, &request[request_cnt]);
	request_cnt++;
	MPI_Irecv(C1_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::right_in], neighbors_id[2], 4, MPI_COMM_WORLD, &request[request_cnt]);
	request_cnt++;

	MPI_Isend(C2_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::up_out], neighbors_id[0], 5, MPI_COMM_WORLD, &request[request_cnt]);
	request_cnt++;
	MPI_Irecv(C2_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::up_in], neighbors_id[0], 6, MPI_COMM_WORLD, &request[request_cnt]);
	request_cnt++;

	MPI_Isend(C2_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::down_out], neighbors_id[3], 6, MPI_COMM_WORLD, &request[request_cnt]);
	request_cnt++;
	MPI_Irecv(C2_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::down_in], neighbors_id[3], 5, MPI_COMM_WORLD, &request[request_cnt]);
	request_cnt++;

	MPI_Isend(C2_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::left_out], neighbors_id[1], 7, MPI_COMM_WORLD, &request[request_cnt]);
	request_cnt++;
	MPI_Irecv(C2_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::left_in], neighbors_id[1], 8, MPI_COMM_WORLD, &request[request_cnt]);
	request_cnt++;

	MPI_Isend(C2_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::right_out], neighbors_id[2], 8, MPI_COMM_WORLD, &request[request_cnt]);
	request_cnt++;
	MPI_Irecv(C2_old.start_pointer(), 1, CMatrix::MPI_Types[(int)direction::right_in], neighbors_id[2], 7, MPI_COMM_WORLD, &request[request_cnt]);
	request_cnt++;

	MPI_Waitall(4 * neighbors_id.size(), request, MPI_STATUS_IGNORE);


	//Calculate the new concentrations for all the points not on the boundary of the domain
	//Note that in parallel the edge of processor's region is not necessarily the edge of the domain
	for (int i = 1; i < imax_local - 1; i++)
		for (int j = 1; j < jmax_local - 1; j++)
		{
			C1.data_2D[i][j] = C1_old.data_2D[i][j] + dt * ((C1_old.data_2D[i][j] * (1.0 - C1_old.data_2D[i][j]) - f * C2_old.data_2D[i][j] * (C1_old.data_2D[i][j] - q) / (C1_old.data_2D[i][j] + q)) / epsilon
				+ D1 * ((C1_old.data_2D[i + 1][j] + C1_old.data_2D[i - 1][j] - 2.0 * C1_old.data_2D[i][j]) / (dx * dx) + (C1_old.data_2D[i][j + 1] + C1_old.data_2D[i][j - 1] - 2.0 * C1_old.data_2D[i][j]) / (dy * dy)));

			C2.data_2D[i][j] = C2_old.data_2D[i][j] + dt * (C1_old.data_2D[i][j] - C2_old.data_2D[i][j]
				+ D2 * ((C2_old.data_2D[i + 1][j] + C2_old.data_2D[i - 1][j] - 2.0 * C2_old.data_2D[i][j]) / (dx * dx) + (C2_old.data_2D[i][j + 1] + C2_old.data_2D[i][j - 1] - 2.0 * C2_old.data_2D[i][j]) / (dy * dy)));
		}

	t += dt;
	delete[] request;
}


void calc_constants() // no change to original code
{
	dx = x_max / ((double)imax - 1);
	dy = y_max / ((double)imax - 1);

	t = 0.0;

	dt = 0.1 * pow(min(dx, dy), 2.0) / (2.0 * max(D1, D2));

	cout << "dt = " << dt << " dx = " << dx << " dy = " << dy << endl;
}


void create_grid() {
	// This function creates the grid which splits the domain across the processors
	// and calculates the index location of the processor on the grid - id_row and id_col

	int difference = p;

	for (int i = 1; i * i <= p; i++) {
		if (p % i == 0) {
			rows = i;
			cols = p / i;
			if (abs(i - p / i) < difference) {
				rows = i;
				cols = p / i;
			}
		}
	}

#ifdef _DEBUG
	if (id == 0) {
		cout << "Grid has " << rows << " rows and " << cols << " columns." << endl;
		cout.flush();
	}
#endif

	id_row = id / cols;
	id_col = id % cols;
}


void do_decomposition() {
	// Create the grid which splits the domain across the processors
	create_grid();

	// Calculate the local grid sizes and start indices
	i_start_local = 0;
	j_start_local = 0;
	int rem_i = imax;
	int rem_j = jmax;

	// Calculate the number of cols each processor will take out of imax rows
	for (int i = 0; i <= id_row; i++) {
		imax_local = rem_i / (rows - i);
		rem_i -= imax_local;
		if (i != id_row) {
			i_start_local += imax_local;
		}
	}

	// Calculate the number of rows each processor will take out of jmax cols
	for (int j = 0; j <= id_col; j++) {
		jmax_local = rem_j / (cols - j);
		rem_j -= jmax_local;
		if (j != id_col) {
			j_start_local += jmax_local;
		}
	}

	// Add padding 
	imax_local += 2;
	jmax_local += 2;
	
	// Adjust the start indices for the padding, might end up with negative indices at the edge processors
	// but that's fine since it's only for the calculation of the initial conditions
	i_start_local--;
	j_start_local--;

#ifdef _DEBUG
	cout << "Processor " << id << " has shape " << "(" << imax_local - 2 << ", " << jmax_local - 2 << ") at " << i_start_local << ", " << j_start_local << endl;
	cout.flush();
#endif

}


void setup_data_profile()
{	
	double origin_x = x_max / 2.0, origin_y = y_max / 2.0;

	// Creates the 2D arrays and sets the initial conditions
	C1.setup_array(imax_local, jmax_local);
	C1_old.setup_array(imax_local, jmax_local);
	C2.setup_array(imax_local, jmax_local);
	C2_old.setup_array(imax_local, jmax_local);

	// Start datatypes for MPI communication
	CMatrix::setup_datatypes(imax_local, jmax_local);
	
	for (int i = 0; i < imax_local; i++)
		for (int j = 0; j < jmax_local; j++)
		{
			// Adgust the x and y coordinates to the local grid using i_start_local and j_start_local
			double x = (i_start_local + i) * dx, y = (j_start_local + j) * dy;
			double angle = atan2(y - origin_y, x - origin_x);			//Note that atan2 is not a square, but arctan taking in the x and y components separately

			if (angle > 0.0 && angle < 0.5)
				C1.data_2D[i][j] = 0.8;
			else
				C1.data_2D[i][j] = q * (f + 1) / (f - 1);

			C2.data_2D[i][j] = q * (f + 1) / (f - 1) + angle / (8 * M_PI * f);
		}
}


void find_neighbours() {
	// Generate the 4 neighbours, always in the same order - up, left, right, down
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			int x = id_row + i;
			int y = id_col + j;
			if (i == j || i == -j) continue;
			if (x < 0) {
				x = rows - 1;
			}
			if (y < 0) {
				y = cols - 1;
			}
			if (x > rows - 1) {
				x = 0;
			}
			if (y > cols - 1) {
				y = 0;
			}
			neighbors_id.push_back(x * cols + y);
		}
	}

	//if(neighbors_id.size() != 4) {
	//	cout << "Error in finding neighbors" << endl;
	//	cout.flush();
	//}

}


int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	srand(time(NULL) + id * 10000);

	auto start = std::chrono::high_resolution_clock::now();

	int out_cnt = 0, it = 0;

	calc_constants();
	do_decomposition();
	setup_data_profile();
	find_neighbours();

#ifdef _DEBUG
	cout << "Processor " << id << " has neighbors: " << endl;
	for (int i = 0; i < neighbors_id.size(); i++) {
		cout << neighbors_id[i] << " ";
	}
	cout << endl;
	cout.flush();
#endif

	grid_to_file(out_cnt);
	out_cnt++;
	t_out += dt_out;

	while (t < t_max)
	{
		//do_iteration_periodic();
		do_iteration_neumann();

		//Note that I am outputing at a fixed time interval rather than after a fixed number of time steps.
		//This means that the output time interval will be independent of the time step (and thus the resolution)
		if (t_out <= t)
		{
			cout << "output: " << out_cnt << "\tt: " << t << "\titeration: " << it << "\tmid " << C1.data_2D[imax / 2][jmax/2] << endl;
			grid_to_file(out_cnt);
			out_cnt++;
			t_out += dt_out;
		}

		it++;
	}


	// Get the end time
	auto end = std::chrono::high_resolution_clock::now();

	// Calculate the elapsed time in milliseconds
	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);

	// Output the elapsed time
	std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

	// Delete the MPI datatypes
	for (int i = 0; i < 8; i++) {
		MPI_Type_free(&CMatrix::MPI_Types[i]);
	}
	MPI_Finalize();

	//Destruct the matrices
	C1.~CMatrix();
	C1_old.~CMatrix();
	C2.~CMatrix();
	C2_old.~CMatrix();

	return 0;
}