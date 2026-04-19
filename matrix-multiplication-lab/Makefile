CXX = g++
CXXFLAGS = -O2 -std=c++17 -fopenmp
TARGET = matrix_mul_omp
SRC = src/main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET) data/result_cpp.txt results/experiments.csv
