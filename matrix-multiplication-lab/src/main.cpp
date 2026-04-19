#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <omp.h>

struct Matrix {
    int n;
    std::vector<long long> data;

    Matrix() : n(0) {}
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

    int n;
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

int main(int argc, char* argv[]) {
    try {
        std::string fileA = "data/matrix_a.txt";
        std::string fileB = "data/matrix_b.txt";
        std::string fileC = "data/result_cpp.txt";
        int requested_threads = omp_get_max_threads();

        if (argc >= 4) {
            fileA = argv[1];
            fileB = argv[2];
            fileC = argv[3];
        }

        if (argc >= 5) {
            requested_threads = std::stoi(argv[4]);
            if (requested_threads <= 0) {
                throw std::runtime_error("Thread count must be positive");
            }
        }

        Matrix A = readMatrix(fileA);
        Matrix B = readMatrix(fileB);

        if (A.n != B.n) {
            throw std::runtime_error("Matrices must be the same size");
        }

        int n = A.n;
        Matrix C(n);

        omp_set_dynamic(0);
        omp_set_num_threads(requested_threads);

        int actual_threads = 1;
        #pragma omp parallel
        {
            #pragma omp single
            actual_threads = omp_get_num_threads();
        }

        auto start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                long long sum = 0;
                for (int k = 0; k < n; ++k) {
                    sum += A.at(i, k) * B.at(k, j);
                }
                C.at(i, j) = sum;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        writeMatrix(fileC, C);

        long long workload = 2LL * n * n * n - 1LL * n * n;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Matrix size: " << n << "x" << n << '\n';
        std::cout << "Threads used: " << actual_threads << '\n';
        std::cout << "Available processors: " << omp_get_num_procs() << '\n';
        std::cout << "Execution time (s): " << duration.count() << '\n';
        std::cout << "Workload (operations): " << workload << '\n';

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
