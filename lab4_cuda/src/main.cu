#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <chrono>
#include <cuda_runtime.h>

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
        throw std::runtime_error("Cannot open output file: " + filename);
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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while (0)

__global__ void matMulKernel(const long long* A, const long long* B, long long* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        long long sum = 0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main(int argc, char* argv[]) {
    try {
        std::string fileA = "data/matrix_a.txt";
        std::string fileB = "data/matrix_b.txt";
        std::string fileC = "data/result_cuda.txt";
        int blockX = 16;
        int blockY = 16;

        if (argc >= 4) {
            fileA = argv[1];
            fileB = argv[2];
            fileC = argv[3];
        }
        if (argc >= 6) {
            blockX = std::stoi(argv[4]);
            blockY = std::stoi(argv[5]);
        }

        Matrix A = readMatrix(fileA);
        Matrix B = readMatrix(fileB);
        if (A.n != B.n) {
            throw std::runtime_error("Matrices must be the same size");
        }

        const int n = A.n;
        const size_t bytes = static_cast<size_t>(n) * n * sizeof(long long);
        Matrix C(n);

        long long* d_A = nullptr;
        long long* d_B = nullptr;
        long long* d_C = nullptr;

        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));

        CUDA_CHECK(cudaMemcpy(d_A, A.data.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B.data.data(), bytes, cudaMemcpyHostToDevice));

        dim3 block(blockX, blockY);
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        matMulKernel<<<grid, block>>>(d_A, d_B, d_C, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        CUDA_CHECK(cudaMemcpy(C.data.data(), d_C, bytes, cudaMemcpyDeviceToHost));

        writeMatrix(fileC, C);

        long long workload = 2LL * n * n * n - 1LL * n * n;

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Matrix size: " << n << "x" << n << '\n';
        std::cout << "Block size: " << blockX << "x" << blockY << '\n';
        std::cout << "Grid size: " << grid.x << "x" << grid.y << '\n';
        std::cout << "Device: " << prop.name << '\n';
        std::cout << "Execution time (ms): " << milliseconds << '\n';
        std::cout << "Execution time (s): " << milliseconds / 1000.0 << '\n';
        std::cout << "Workload (operations): " << workload << '\n';

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}