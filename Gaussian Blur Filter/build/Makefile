# Makefile for CUDA Gaussian Blur project

CC = nvcc
CFLAGS = -O2
LDFLAGS = `pkg-config --cflags --libs opencv4`

TARGET = gaussian_blur
SRC = ../src/gaussian_blur.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(SRC) -o $(TARGET) $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(TARGET)
