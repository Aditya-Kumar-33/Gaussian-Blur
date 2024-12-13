# CUDA Gaussian Blur Filter

## Description
This project applies a Gaussian blur filter to an image using CUDA for GPU acceleration. The image processing is done in parallel on the GPU, significantly speeding up the process compared to traditional CPU-based methods.

## Requirements
- CUDA toolkit
- OpenCV library (`pip install opencv-python`)

## Installation
1. Install the CUDA toolkit: https://developer.nvidia.com/cuda-downloads
2. Install OpenCV:
   ```bash
   pip install opencv-python

### How to Run

    Place your image (e.g., input.jpg) in the project directory.
    Compile the code using the nvcc compiler:

    make

    Run the program:

    ./gaussian_blur

### Output
The output image will be saved as output.jpg.