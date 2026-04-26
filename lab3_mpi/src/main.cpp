#include <mpi.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Matrix {
    int n = 0;
    std::vector<long long> data;

    Matrix() = default;
    explicit Matrix(int size) : n(size), data(static_cast<size_t>(size) * size, 0) {}

    long long& at(int i, int j) {
        return data[static_cast<size_t>(i) * n + j];
    }

    const long long& at(int i, int j) const {
        return data[static_cast<size_t>(i) * n + j];
    }
};

Matrix readMatrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    int n = 0;
    file >> n;
    if (!file || n <= 0) {
        throw std::runtime_error("Invalid matrix size in file: " + filename);
    }

    Matrix matrix(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (!(file >> matrix.at(i, j))) {
                throw std::runtime_error("Invalid matrix data in file: " + filename);
            }
        }
    }

    return matrix;
}

void writeMatrix(const std::string& filename, const Matrix& matrix) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot write to file: " + filename);
    }

    file << matrix.n << '\n';
    for (int i = 0; i < matrix.n; ++i) {
        for (int j = 0; j < matrix.n; ++j) {
            file << matrix.at(i, j);
            if (j + 1 < matrix.n) {
                file << ' ';
            }
        }
        file << '\n';
    }
}

std::vector<int> buildRowCounts(int n, int worldSize) {
    std::vector<int> rows(worldSize, n / worldSize);
    for (int i = 0; i < n % worldSize; ++i) {
        rows[i] += 1;
    }
    return rows;
}

std::vector<int> buildElementCounts(const std::vector<int>& rows, int n) {
    std::vector<int> counts(rows.size(), 0);
    for (size_t i = 0; i < rows.size(); ++i) {
        counts[i] = rows[i] * n;
    }
    return counts;
}

std::vector<int> buildDisplacements(const std::vector<int>& counts) {
    std::vector<int> displs(counts.size(), 0);
    for (size_t i = 1; i < counts.size(); ++i) {
        displs[i] = displs[i - 1] + counts[i - 1];
    }
    return displs;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int worldSize = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    std::string fileA = "data/matrix_a.txt";
    std::string fileB = "data/matrix_b.txt";
    std::string fileC = "data/result_mpi.txt";

    if (argc >= 4) {
        fileA = argv[1];
        fileB = argv[2];
        fileC = argv[3];
    }

    Matrix A;
    Matrix B;
    Matrix C;
    int n = 0;
    int status = 0;
    std::string errorMessage;

    if (rank == 0) {
        try {
            A = readMatrix(fileA);
            B = readMatrix(fileB);
            if (A.n != B.n) {
                throw std::runtime_error("Matrices must be the same size");
            }
            n = A.n;
            C = Matrix(n);
        } catch (const std::exception& e) {
            status = 1;
            errorMessage = e.what();
        }
    }

    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status != 0) {
        if (rank == 0) {
            std::cerr << "Error: " << errorMessage << '\n';
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        B = Matrix(n);
    }

    std::vector<int> rowsPerProcess = buildRowCounts(n, worldSize);
    std::vector<int> sendCounts = buildElementCounts(rowsPerProcess, n);
    std::vector<int> displacements = buildDisplacements(sendCounts);

    int localRows = rowsPerProcess[rank];
    std::vector<long long> localA(static_cast<size_t>(localRows) * n, 0);
    std::vector<long long> localC(static_cast<size_t>(localRows) * n, 0);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Bcast(B.data.data(), n * n, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    MPI_Scatterv(
        rank == 0 ? A.data.data() : nullptr,
        sendCounts.data(),
        displacements.data(),
        MPI_LONG_LONG,
        localA.data(),
        localRows * n,
        MPI_LONG_LONG,
        0,
        MPI_COMM_WORLD
    );

    for (int i = 0; i < localRows; ++i) {
        for (int j = 0; j < n; ++j) {
            long long sum = 0;
            for (int k = 0; k < n; ++k) {
                sum += localA[static_cast<size_t>(i) * n + k] * B.at(k, j);
            }
            localC[static_cast<size_t>(i) * n + j] = sum;
        }
    }

    MPI_Gatherv(
        localC.data(),
        localRows * n,
        MPI_LONG_LONG,
        rank == 0 ? C.data.data() : nullptr,
        sendCounts.data(),
        displacements.data(),
        MPI_LONG_LONG,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        try {
            writeMatrix(fileC, C);
            long long workload = 2LL * n * n * n - 1LL * n * n;

            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Matrix size: " << n << "x" << n << '\n';
            std::cout << "MPI processes: " << worldSize << '\n';
            std::cout << "Execution time (s): " << elapsed << '\n';
            std::cout << "Workload (operations): " << workload << '\n';
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << '\n';
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
