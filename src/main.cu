#include <iostream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cuda_runtime.h>
#define THREADS_PER_BLOCK_x 768
#define THREADS_PER_BLOCK_y 1

using namespace std;
using namespace cv;

//Size of image
__const__ int WIDTH=1920;
//__const__ int HIGHT=1277;

//Number of iteration
__const__ int ite=15;
__const__ int s = ite/2;

#define CHECK(call){                                                    \
   const cudaError_t error = call;                                      \
   if (error != cudaSuccess){                                           \
	printf("Error: %s:%d,  ", __FILE__, __LINE__);                      \
	printf("code:%d, reason: %s\n", error,cudaGetErrorString(error));   \
    exit(1);                                                            \
   }                                                                    \
 }                                                                      \

__global__ void calc(unsigned char* device,unsigned char* device_r) {

	//calculate position
    int xid=threadIdx.x+blockIdx.x*blockDim.x;
    int yid=threadIdx.y+blockIdx.y*blockDim.y;
    int idx=yid*WIDTH+xid;

    //filtering
    float sum = 0;
    for(int x=-s;x<=s;x++){
    	for(int y=-s;y<=s;y++){
         sum = sum + device[x+xid+(y+yid)*WIDTH];
        }
    }
	device_r[idx]=sum/(ite*ite);
}

void gpu(Mat& img){

	unsigned char* host;//pointer on host for input
	unsigned char* host_r;//pointer on host for output

	unsigned char* device;//pointer on device for input
	unsigned char* device_r;//pointer on device for output

	Mat img_r = Mat(img.size(), img.type());

	//refer data by using pointer
	host = img.data;
	host_r=img_r.data;;

	//calculate amount of memory
	const int cols=img.cols;
	const int rows=img.rows;
	const int N =cols*rows;
	size_t n_byte = N * sizeof(unsigned char);

    //allocate memory
    CHECK(cudaMalloc((void**)&device, N));
    CHECK(cudaMalloc((void**)&device_r, N));

	//copy data to device from host
	CHECK(cudaMemcpy(device, host, n_byte, cudaMemcpyHostToDevice));

	//calculation in GPU
	dim3 Block_dim(THREADS_PER_BLOCK_x,THREADS_PER_BLOCK_y);
	dim3 Grid_dim((cols+Block_dim.x-1)/Block_dim.x,(rows+Block_dim.y-1)/Block_dim.y);

	cout<<"number of thread : {"<<N<<"}"<<endl;
	printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
	Grid_dim.x, Grid_dim.y, Grid_dim.z, Block_dim.x, Block_dim.y, Block_dim.z);

	calc<<<Grid_dim,Block_dim>>>(device,device_r);
	CHECK(cudaDeviceSynchronize());

	//Check kernel error
	CHECK(cudaGetLastError());

	//copy result to host from device
	CHECK(cudaMemcpy(host_r,device_r, n_byte, cudaMemcpyDeviceToHost));

	//show and save result image
	imwrite("../img/output_gpu.jpg",img_r);
	imshow("Output",img_r);
	cvWaitKey(0);

	//outsource memory
	CHECK(cudaFree(device));
	CHECK(cudaFree(device_r));

	free(host);
    free(host_r);
    cudaDeviceReset();
}

void cpu(Mat& img){
	float sum;
	Mat img_r = Mat(img.size(), img.type());
	for(int v=0;v<img.rows;v++){
		for(int u=0;u<img.cols;u++){
			sum = 0;
			for(int x=-s;x<=s;x++){
				for(int y=-s;y<=s;y++){
					sum=sum+img.at<unsigned char>(v+y,u+x);
				}
			}
			img_r.at<unsigned char>(v,u)=sum/(ite*ite);
		}
	}
	//show and save result image
	imwrite("../img/output_cpu.jpg",img_r);
	imshow("Output",img_r);
	cvWaitKey(0);
}

int main(void){

	//read image
	Mat img=imread("../img/input.jpg", IMREAD_GRAYSCALE);

	//GPU-Processing
	//gpu(img);

	//CPU-Processing
	cpu(img);

return 0;
}
