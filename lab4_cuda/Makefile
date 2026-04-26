NVCC = nvcc
NVCCFLAGS = -O2 -std=c++17
TARGET = matrix_mul_cuda
SRC = src/main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET) data/result_cuda.txt

run:
	./$(TARGET) data/matrix_a.txt data/matrix_b.txt data/result_cuda.txt 16 16