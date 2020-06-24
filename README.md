# Image_processing_CUDA
<table>
  <tr>
    <td><img src="img/input.jpg"></td>
    <td><img src="img/output_gpu.jpg"></td>
  </tr>
  <tr>
    <td align="middle">Input Image</td>
    <td align="middle">Output Image After Median Blur</td>
  </tr>
</table>

# Build and Execution

## Dependencies

  * CMake 2.8.11 or higher.
  * Cuda 7.5 or higher.
  * GCC 4.8.
  * OpenCV 3.1 or higher.

## Build (Linux)

    git clone https://github.com/nagaitomoaki/Image_processing_CUDA.git
    mkdir build
    cd build
    cmake ..
    make
    ./main
