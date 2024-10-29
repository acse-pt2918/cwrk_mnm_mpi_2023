#pragma once

#include <mpi.h>
#include <vector>

using std::vector;

enum class direction {
    up_in,
    up_out,
    down_in,
    down_out,
    left_in,
    left_out,
    right_in,
    right_out
};

class CMatrix {
public:
    double* data_1D;
    double** data_2D;
    int i_max, j_max;

    CMatrix();
    ~CMatrix();

    void setup_array(int height, int width);
    double* start_pointer();
    void swap_data(CMatrix& other);

    static MPI_Datatype MPI_Types[8];
    static void setup_datatypes(int height, int width);
};

