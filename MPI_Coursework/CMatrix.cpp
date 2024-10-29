#include "CMatrix.h"

CMatrix::CMatrix() {
    i_max = 0;
    j_max = 0;
    data_1D = nullptr;
    data_2D = nullptr;
}

CMatrix::~CMatrix() {
    delete[] data_1D;
    delete[] data_2D;
}

void CMatrix::setup_array(int height, int width) {
    j_max = width;
    i_max = height;
    data_1D = new double[width * height];
    data_2D = new double* [i_max];

    for (int i = 0; i < i_max; i++)
        data_2D[i] = &data_1D[i * j_max];

    //Set array to zero - temp - might need to remove
    for (int i = 0; i < i_max; i++)
		for (int j = 0; j < j_max; j++)
			data_2D[i][j] = 0.0;
}

double* CMatrix::start_pointer() {
    return data_1D;
}

void CMatrix::swap_data(CMatrix& other) {
    // Swap data_1D pointers
    double* temp_1D = this->data_1D;
    this->data_1D = other.data_1D;
    other.data_1D = temp_1D;

    // Swap data_2D pointers
    double** temp_2D = this->data_2D;
    this->data_2D = other.data_2D;
    other.data_2D = temp_2D;
}

MPI_Datatype CMatrix::MPI_Types[8];

void CMatrix::setup_datatypes(int height, int width) {
    vector<MPI_Datatype> type_list;
    vector<int> block_lengths;
    vector<MPI_Aint> displacements;

    MPI_Aint address, start_address;

    CMatrix temp;
    temp.setup_array(height, width);

    MPI_Get_address(temp.data_1D, &start_address);

    // Up received
    type_list.push_back(MPI_DOUBLE);
    block_lengths.push_back(width - 2);
    MPI_Get_address(&temp.data_2D[0][1], &address);
    displacements.push_back(address);

    for (int i = 0; i < displacements.size(); i++)
        displacements[i] -= start_address;

    MPI_Type_create_struct(displacements.size(), block_lengths.data(), displacements.data(), type_list.data(), &MPI_Types[(int)direction::up_in]);
    MPI_Type_commit(&MPI_Types[(int)direction::up_in]);

    type_list.clear();
    block_lengths.clear();
    displacements.clear();

    // Up sent
    type_list.push_back(MPI_DOUBLE);
    block_lengths.push_back(width - 2);
    MPI_Get_address(&temp.data_2D[1][1], &address);
    displacements.push_back(address);

    for (int i = 0; i < displacements.size(); i++)
        displacements[i] -= start_address;

    MPI_Type_create_struct(displacements.size(), block_lengths.data(), displacements.data(), type_list.data(), &MPI_Types[(int)direction::up_out]);
    MPI_Type_commit(&MPI_Types[(int)direction::up_out]);

    type_list.clear();
    block_lengths.clear();
    displacements.clear();

    // Down received
    type_list.push_back(MPI_DOUBLE);
    block_lengths.push_back(width - 2);
    MPI_Get_address(&temp.data_2D[height - 1][1], &address);
    displacements.push_back(address);

    for (int i = 0; i < displacements.size(); i++)
        displacements[i] -= start_address;

    MPI_Type_create_struct(displacements.size(), block_lengths.data(), displacements.data(), type_list.data(), &MPI_Types[(int)direction::down_in]);
    MPI_Type_commit(&MPI_Types[(int)direction::down_in]);

    type_list.clear();
    block_lengths.clear();
    displacements.clear();

    // Down sent out
    type_list.push_back(MPI_DOUBLE);
    block_lengths.push_back(width - 2);
    MPI_Get_address(&temp.data_2D[height - 2][1], &address);
    displacements.push_back(address);

    for (int i = 0; i < displacements.size(); i++)
        displacements[i] -= start_address;

    MPI_Type_create_struct(displacements.size(), block_lengths.data(), displacements.data(), type_list.data(), &MPI_Types[(int)direction::down_out]);
    MPI_Type_commit(&MPI_Types[(int)direction::down_out]);

    type_list.clear();
    block_lengths.clear();
    displacements.clear();

    // Left received
    for (int i = 1; i < height - 1; i++) {
        type_list.push_back(MPI_DOUBLE);
        block_lengths.push_back(1);
        MPI_Get_address(&temp.data_2D[i][0], &address);
        displacements.push_back(address);
    }

    for (int i = 0; i < displacements.size(); i++)
        displacements[i] -= start_address;

    MPI_Type_create_struct(displacements.size(), block_lengths.data(), displacements.data(), type_list.data(), &MPI_Types[(int)direction::left_in]);
    MPI_Type_commit(&MPI_Types[(int)direction::left_in]);

    type_list.clear();
    block_lengths.clear();
    displacements.clear();

    // Left sent
    for (int i = 1; i < height - 1; i++) {
        type_list.push_back(MPI_DOUBLE);
        block_lengths.push_back(1);
        MPI_Get_address(&temp.data_2D[i][1], &address);
        displacements.push_back(address);
    }

    for (int i = 0; i < displacements.size(); i++)
        displacements[i] -= start_address;

    MPI_Type_create_struct(displacements.size(), block_lengths.data(), displacements.data(), type_list.data(), &MPI_Types[(int)direction::left_out]);
    MPI_Type_commit(&MPI_Types[(int)direction::left_out]);

    type_list.clear();
    block_lengths.clear();
    displacements.clear();

    // Right received
    for (int i = 1; i < height - 1; i++) {
        type_list.push_back(MPI_DOUBLE);
        block_lengths.push_back(1);
        MPI_Get_address(&temp.data_2D[i][width - 1], &address);
        displacements.push_back(address);
    }

    for (int i = 0; i < displacements.size(); i++)
        displacements[i] -= start_address;

    MPI_Type_create_struct(displacements.size(), block_lengths.data(), displacements.data(), type_list.data(), &MPI_Types[(int)direction::right_in]);
    MPI_Type_commit(&MPI_Types[(int)direction::right_in]);

    type_list.clear();
    block_lengths.clear();
    displacements.clear();

    // Right sent out
    for (int i = 1; i < height - 1; i++) {
        type_list.push_back(MPI_DOUBLE);
        block_lengths.push_back(1);
        MPI_Get_address(&temp.data_2D[i][width - 2], &address);
        displacements.push_back(address);
    }

    for (int i = 0; i < displacements.size(); i++)
        displacements[i] -= start_address;

    MPI_Type_create_struct(displacements.size(), block_lengths.data(), displacements.data(), type_list.data(), &MPI_Types[(int)direction::right_out]);
    MPI_Type_commit(&MPI_Types[(int)direction::right_out]);

    type_list.clear();
    block_lengths.clear();
    displacements.clear();
}
