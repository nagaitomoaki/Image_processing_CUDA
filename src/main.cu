#include <iostream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cuda_runtime.h>
#define THREADS_PER_BLOCK 32

using namespace std;
using namespace cv;

#define CHECK(call){                                                    \
   const cudaError_t error = call;                                      \
   if (error != cudaSuccess){                                           \
	printf("Error: %s:%d,  ", __FILE__, __LINE__);                      \
	printf("code:%d, reason: %s\n", error,cudaGetErrorString(error));   \
    exit(1);                                                            \
   }                                                                    \
 }                                                                      \

__global__ void calc(unsigned char* device,unsigned char* device_r) {
    int xid=threadIdx.x+blockIdx.x*blockDim.x;
    int yid=threadIdx.y+blockIdx.y*blockDim.y;
    int idx=yid*1920+xid;
	device_r[idx]=device[idx]*0.5;
}

void gpu(Mat& img){

	unsigned char* host;//ポインタ定義//ホスト上のポインタ
	unsigned char* host_r;//ポインタ定義//ホスト上のポインタ

	unsigned char* device;//デバイス上のポインタ
	unsigned char* device_r;//デバイス上の結果

	Mat img_r = Mat(img.size(), img.type());

	//ポインタで参照
	host = img.data;//imgdataのポインタってアドレス？？だからcharで受け取る？
	host_r=img_r.data;;

	//メモリ量計算
	const int cols=img.cols;
	const int rows=img.rows;
	const int N =cols*rows;
	size_t n_byte = N * sizeof(unsigned char);

    //cudaにメモリ領域割当
    CHECK(cudaMalloc((void**)&device, N));
    CHECK(cudaMalloc((void**)&device_r, N));

	//cudaメモリに値コピー
	CHECK(cudaMemcpy(device, host, n_byte, cudaMemcpyHostToDevice));//多めにコピーしてるけどいいか

	//gpu計算
	dim3 Block_dim(THREADS_PER_BLOCK,THREADS_PER_BLOCK);//(16,16,1)
	dim3 Grid_dim((cols+Block_dim.x-1)/Block_dim.x,(rows+Block_dim.y-1)/Block_dim.y);//(129600,1,1)

	cout<<"number of thread : {"<<N<<"}"<<endl;
	printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
	Grid_dim.x, Grid_dim.y, Grid_dim.z, Block_dim.x, Block_dim.y, Block_dim.z);

	calc<<<Grid_dim,Block_dim>>>(device,device_r);//grid(いつくブロックが必要か),block(ブロックの大きさ:スレッド×スレッド)
	CHECK(cudaDeviceSynchronize());//CPUが先に進むのを防ぐ

	//カーネルのエラーをチェック
	CHECK(cudaGetLastError());

	//計算結果をホスト側にコピー
	CHECK(cudaMemcpy(host_r,device_r, n_byte, cudaMemcpyDeviceToHost));

	/*結果表示*/
	imwrite("img/output.jpg",img_r);
	imshow("img",img_r);
	cvWaitKey(0);

	//メモリ解放
	CHECK(cudaFree(device));
	CHECK(cudaFree(device_r));
	//free(host);
    //free(host_r);
    cudaDeviceReset();
}

void cpu(Mat& img){
	Mat img_r = Mat(img.size(), img.type());
	for(int v=0;v<img.rows;v++){
		for(int u=0;u<img.cols;u++){
			img_r.at<unsigned char>(v,u)=img.at<unsigned char>(v,u)*0.5;
		}
	}
}

int main(void){

	//read image
	Mat img=imread("img/aragaki.jpg",0);

	//GPU-Processing
	gpu(img);

	//CPU-Processing
	//cpu(img);

return 0;
}
