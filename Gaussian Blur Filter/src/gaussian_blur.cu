#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

__global__ void gaussianBlurKernel(unsigned char *inputImage, unsigned char *outputImage, int width, int height, float *kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        int halfKernel = kernelSize / 2;

        for (int ky = -halfKernel; ky <= halfKernel; ky++) {
            for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                int nx = min(max(x + kx, 0), width - 1);
                int ny = min(max(y + ky, 0), height - 1);
                sum += inputImage[ny * width + nx] * kernel[(ky + halfKernel) * kernelSize + (kx + halfKernel)];
            }
        }
        outputImage[y * width + x] = static_cast<unsigned char>(sum);
    }
}

void applyGaussianBlur(const cv::Mat &inputImage, cv::Mat &outputImage) {
    int width = inputImage.cols;
    int height = inputImage.rows;

    // Define a Gaussian kernel (3x3 for simplicity)
    float h_kernel[9] = {
        0.0625f, 0.125f, 0.0625f,
        0.125f, 0.25f, 0.125f,
        0.0625f, 0.125f, 0.0625f
    };

    float *d_kernel;
    unsigned char *d_input, *d_output;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_input, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_kernel, 9 * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_input, inputImage.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, 9 * sizeof(float), cudaMemcpyHostToDevice);

    // Set up the grid and block dimensions
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, 1);

    // Launch the kernel
    gaussianBlurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, d_kernel, 3);

    // Copy the result back to CPU
    cudaMemcpy(outputImage.data, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

int main() {
    // Load the image
    cv::Mat inputImage = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    cv::Mat outputImage(inputImage.size(), inputImage.type());

    // Apply Gaussian Blur using CUDA
    applyGaussianBlur(inputImage, outputImage);

    // Save the output image
    cv::imwrite("output.jpg", outputImage);

    std::cout << "Gaussian blur applied successfully!" << std::endl;
    return 0;
}
