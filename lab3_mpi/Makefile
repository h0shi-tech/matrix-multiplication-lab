CXX = mpic++
CXXFLAGS = -O2 -std=c++17
TARGET = matrix_mul_mpi
SRC = src/main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET) data/result_mpi.txt
