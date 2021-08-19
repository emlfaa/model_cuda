#include "cudamn.h"

#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <vector_functions.h>
#include <math.h>
#include <time.h>

#define THREAD_DIM_X 512
__device__ double gausArray_gpu[15*15];
//#define PI 3.1415926
#define CLOCKS_PER_SEC ((clock_t)1000000)

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////                                   cuda显存池                                                    //////
///////////////////////////////////////////////////////////////////////////////////////////////////////////

GpuMem_message::GpuMem_message()
{

}

bool GpuMem_message::create(size_t _size) {
    cudaMalloc((void**)&cuda_addr, _size);
    if(cuda_addr == NULL){
        printf("create cuda memory is wrong, size = %ld\n", _size);
        return false;
    }
    cuda_size = _size;
    usable = true;
    ever_used = false;
    return true;
}

void GpuMem_message::destroy() {
    cudaFree(cuda_addr);
    cuda_size = 0;
    usable = false;
    ever_used = false;
    return;
}

Gpu_Mem::Gpu_Mem()
{

}

bool Gpu_Mem::init_Gpu_Mem(const int width, const int height, const int sc)
{
    imgW = width;
    imgH = height;

    int size_uint8_t = sizeof(uint8_t); // 1
    int size_uchar3 = sizeof(uchar3); // 3

    size_t size = width * height;

    std::vector<size_t> sizes = {
            256, 256, 256, 256, 256,
            size * 3 / sc / sc,
            size, size, size,
            size * size_uint8_t, size * size_uint8_t, size * size_uint8_t, size * size_uint8_t, size * size_uint8_t, size * size_uint8_t,
            size * size_uint8_t, size * size_uint8_t, size * size_uint8_t, size * size_uint8_t,
            size * size_uchar3,
            size * 3,
    };

    gpu_mems.resize(sizes.size());
    for(int i = 0; i, i < gpu_mems.size(); i++){
        gpu_mems[i].create(sizes[i]);
    }
}

bool Gpu_Mem::destroy_Gpu_Mem() {
    for(int i = 0; i < gpu_mems.size(); i++)
    {
        if(!gpu_mems[i].ever_used){
            printf("!!!!!! GPU MEM size==%d never been used!!! \n", gpu_mems[i].cuda_size);
        }
        if (!gpu_mems[i].usable) {
            printf("there is gpu_mems not returned , size = %d\n", gpu_mems[i].cuda_size);
        }
        gpu_mems[i].destroy();
    }
    assert(gpu_mems.empty());
    return true;

}

bool Gpu_Mem::createGpuMemory(void** memobj, size_t _size)
{
    for(int i = 0; i < gpu_mems.size(); i++){
        if(gpu_mems[i].cuda_size >= _size && gpu_mems[i].usable)
        {

            gpu_mems[i].usable = false;
            gpu_mems[i].ever_used = true;
            *memobj = gpu_mems[i].cuda_addr;
            return true;
        }

    }


    return NULL;
}
bool Gpu_Mem::releaseGpuMemory(uint8_t *memobj)
{
    for(int i = 0; i < gpu_mems.size(); i++){
        if(gpu_mems[i].cuda_addr == memobj){
            gpu_mems[i].usable = true;
            return true;
        }
    }
    return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////                                   cuda函数                                                     //////
///////////////////////////////////////////////////////////////////////////////////////////////////////////


inline __device__ __host__ float regin_union(float a1, float a2, float b1, float b2) {
    return (min)(a2, b2) - (max)(a1, b1);
}

// 下采样图片
__global__ void _resize_n8u(uint8_t *src_pix, int src_w, int src_h, int nc, uint8_t *dst_px,
                            int dst_w, int dst_h){
    const int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dc = gidx % nc, dx = gidx / nc % dst_w, dy = gidx / nc / dst_w;
    if(dy >= dst_h) return;
    const float bw = (float) src_w / (float) dst_w, bh = (float) src_h / (float) dst_h;
    const float fx1 = dx * bw, fx2 = (dx + 1) * bw, fy1 = dy * bh, fy2 = (dy + 1) * bh;
    const int st_x = (int) floor(fx1), ed_x = (int) ceil(fx2);
    const int st_y = (int) floor(fy1), ed_y = (int) ceil(fy2);
    float sv = 0;
    for(int x = st_x; x < ed_x; x++){
        float px = regin_union(x, x + 1.0f, fx1, fx2);
        for(int y = st_y; y < ed_y; y++){
            float py = regin_union(y, y + 1.0f, fy1, fy2);
            sv += src_pix[nc * (x + src_w * y) + dc] * px * py;
        }
    }
    sv /= bw * bh;
    dst_px[nc * (dy * dst_w + dx) + dc] = (uint8_t) sv;

}

// RGB to gray
__global__ void bgr2gray_kernel(const uint8_t * bgr_img, int h, int w, uint8_t *gray){
    int indx = blockIdx.x * blockDim.x + threadIdx.x;
    if(indx > h * w) return;
    const uint8_t *bgr = bgr_img + 3 * indx;
    gray[indx] = (uint8_t)(0.114f * bgr[0] + 0.587f * bgr[1] + 0.299f * bgr[2]);
}

// 彩色图像变为灰度kernel 方法二
// https://blog.csdn.net/lingsuifenfei123/article/details/83444159
__global__ void _rgb2gray(uchar3 * const d_in, int imgheight, int imgwidth, unsigned char *d_out)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < imgwidth && idy < imgheight)
    {
        uchar3 rgb = d_in[idy * imgwidth + idx];
        d_out[idy * imgwidth + idx] = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
    }
}

// 调整图像灰度等级kernel
__global__ void _gray_level(uint8_t *src_pix, int src_w, int src_h, int nc, uint8_t *dst_px, int level)
{
    const int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(gidx >= src_h * src_w)
        return;

    int block_num = level - 1;
    int block_size = 256 / block_num;
    int i = 1;
    for(i; i <= block_num; i++)
    {
        if(src_pix[gidx] > block_size * i)
            continue;

        int mid_value = block_size * i / 2;
        int left = block_size * (i - 1);
        int right = block_size * i - 1;
        if(src_pix[gidx] < mid_value)
        {
            dst_px[gidx] = left;
            break;
        }
        else
        {
            dst_px[gidx] = right;
            break;
        }

    }

    if(i > block_num)
        dst_px[gidx] = src_pix[gidx];


}

// 统计图像中所有的像素值的数值
__global__ void _cuSumPix(uint8_t* h_in, int* nSumPixGpu, int img_w, int img_h){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= img_h * img_w)
        return;

    int line_number = tid / img_w;
    int the_number = tid % img_w;

    atomicAdd(&nSumPixGpu[(int)h_in[line_number * img_w + the_number]], 1);

}

__global__ void _cuSumPix2(uint8_t* h_in, int* nSumPixGpu, int img_w, int img_h){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
//    printf("imgw: %d, imgh: %d\n", img_w, img_h);
//    printf("\n");
//    printf("tid = %d\n", tid);
//    printf("\n");
    for(int i = 0; i < 256; i++)
    {
        printf("nSumPixGpu[%d]: %d \n", i, nSumPixGpu[i]);
    }

//    if(tid >= img_h * img_w)
//        return;
//
//    int line_number = tid / img_w;
//    int the_number = tid % img_w;
//
//    atomicAdd(&nSumPixGpu[(int)h_in[line_number * img_w + the_number]], 1);
//    if(tid == 1)
//        for(int i = 0; i < 256; i++)
//        {
//            printf("nSumPixGpu[%d]: %d \n", i, nSumPixGpu[i]);
//        }
}

// 图像均衡化
__global__ void _cugrayequalized(uint8_t* h_in, uint8_t* h_out, int* mgpuEqua, int imageSize_gray1)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid > imageSize_gray1)
        return;

    h_out[tid] = mgpuEqua[h_in[tid]];
}
//__global__ void _cugrayequalized(uint8_t* h_in, uint8_t* h_out, int* mgpuEqua, int imgW, int imgH)
//{
//    const unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
//    const unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
//
//    if(col < imgW &&row < imgH)
//    {
//        const unsigned long pid =  imgW * row + col;
//        h_out[pid] = mgpuEqua[h_in[pid]];
//    }
//}

// 灰度图像取反操作
__global__ void _PixelNegate(uint8_t* img_in, uint8_t* img_out, int imagesize){

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid > imagesize)
        return;

    img_out[tid] = 255 - img_in[tid];
}

__global__ void _Gaussian(uint8_t* img_in, uint8_t* img_out, int size, int imagesize, int imgW, int imgH, int sum){

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid > imagesize)
        return;

    int i = blockIdx.x;    // 当前像素的高坐标
    int j = threadIdx.x;   // 当前像素的宽坐标
    int k = 0;
    int temp = 0;
    for(int l = -size/2; l <= size/2; l++)
    {
        for(int g = -size/2; g <= size/2; g++)
        {
            int row = i + l;
            int col = j + g;
            row = row<0?0:row;
            row = row>=imgH?imgH-1:row;
            col = col<0?0:col;
            col=col>=imgW?imgW-1:col;
            temp = gausArray_gpu[k] * img_in[row*col];
            k++;
        }
    }

    if(temp / sum < 0)
        img_out[tid] = 0;
    else if(temp / sum > 255)
        img_out[tid] = 255;
    else
        img_out[tid] = temp / sum;
}

__global__ void gaussianFilter(uint8_t *d_in, uint8_t *d_out, int width, int height, int allsum)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;


    int sum = 0;
    int index = 0;


    if (tidx>2 && tidx<width - 2 && tidy>2 && tidy<height - 2)
        return;

    for (int m = tidx - 2; m < tidx + 3; m++)
    {
        for (int n = tidy - 2; n < tidy + 3; n++)
        {
            sum += d_in[m*width + n] * gausArray_gpu[index++];
        }
    }
    if (sum / allsum<0)
        *(d_out + (tidx)*width + tidy) = 0;
//        d_out[tidx * width + tidy] = 0;
    else if (sum / allsum>255)
        *(d_out + (tidx)*width + tidy) = 255;
//        d_out[tidx * width + tidy] = 255;
    else
        *(d_out + (tidx)*width + tidy) = sum / allsum;
//        d_out[tidx * width + tidy] = sum / allsum;
}


// 颜色减淡混合
__global__ void _ColorDodgeBlend(uint8_t* gray, uint8_t* blur, uint8_t* img_out, int scale, int imagesize){

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid > imagesize)
        return;

    img_out[tid] = gray[tid] / (255 - blur[tid]) * 255;
//    img_out[tid] = blur[tid];
}

// https://blog.csdn.net/mgotze/article/details/77934138
// Sobel算子边缘检测核函数
__global__ void sobelInCuda(uint8_t *dataIn, uint8_t *dataOut, int imgHeight, int imgWidth){
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    int index = yIndex * imgWidth + xIndex;
    int Gx = 0;
    int Gy = 0;

    if (xIndex > 0 && xIndex < imgWidth - 1 && yIndex > 0 && yIndex < imgHeight - 1)
    {
        Gx = dataIn[(yIndex - 1) * imgWidth + xIndex + 1] + 2 * dataIn[yIndex * imgWidth + xIndex + 1] + dataIn[(yIndex + 1) * imgWidth + xIndex + 1]
             - (dataIn[(yIndex - 1) * imgWidth + xIndex - 1] + 2 * dataIn[yIndex * imgWidth + xIndex - 1] + dataIn[(yIndex + 1) * imgWidth + xIndex - 1]);
        Gy = dataIn[(yIndex - 1) * imgWidth + xIndex - 1] + 2 * dataIn[(yIndex - 1) * imgWidth + xIndex] + dataIn[(yIndex - 1) * imgWidth + xIndex + 1]
             - (dataIn[(yIndex + 1) * imgWidth + xIndex - 1] + 2 * dataIn[(yIndex + 1) * imgWidth + xIndex] + dataIn[(yIndex + 1) * imgWidth + xIndex + 1]);
        dataOut[index] = (abs(Gx) + abs(Gy)) / 2;
    }
}

// 二值化
__global__ void _binarizationCuda(uint8_t *d_in, uint8_t *d_out, int threshold, int imgW, int imgH){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= imgH * imgW)
        return;

    int line_number = tid / imgW;
    int the_number = tid % imgW;

    if(d_in[line_number * imgW + the_number] >= threshold)
        d_out[line_number * imgW + the_number] = 255;
    else
        d_out[line_number * imgW + the_number] = 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////                                    工具函数                                                     //////
///////////////////////////////////////////////////////////////////////////////////////////////////////////


// https://www.cnblogs.com/mtcnn/p/9411978.html
// ******************高斯卷积核生成函数*************************
void img_cudaprocess::GetGaussianKernel(double **gaus, int size, double sigma, double& temp){
//    //方法一
    const double PI =4.0 * atan(1.0);  // 圆周率π赋值
    int center = size / 2;
    double sum = 0;


    for(int i = 0; i < size; i++)
    {

        for(int j = 0; j < size; j++)
        {
//            std::cout<<"eeeeeeeeesssaee"<<std::endl;
            gaus[i][j]=(1/(2*PI*sigma*sigma))*exp(-((i-center)*(i-center)+(j-center)*(j-center))/(2*sigma*sigma));
            sum += gaus[i][j];
//            std::cout<<gaus[i][j]<<std::endl;
        }
    }

    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++)
        {
            gaus[i][j] /= sum;
//            std::cout<<gaus[i][j]<<"  ";
        }
//        std::cout<<std::endl<<std::endl;
    }

    double k = 1 / gaus[0][0];
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            gaus[i][j] *= k;
            temp += gaus[i][j];
        }
    }
//    std::exit(0);

    // 方法二
//    int center = size / 2;
//    double x2, y2;
//    for(int i = 0; i < size; i++){
//        x2 = pow(i - center, 2);
//        for(int j = 0; j < size; j++){
//            y2 = pow(j - center, 2);
//            double g = exp(-(x2 +y2)) / (2 * sigma * sigma);
//            g /= 2 * PI * sigma;
//            gaus[i][j] = g;
//        }
//    }
//
//    double k = 1 / gaus[0][0];
//    for(int i = 0; i < size; i++) {
//        for (int j = 0; j < size; j++) {
//            gaus[i][j] *= k;
//        }
//    }

    return;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////                                    C++函数                                                     //////
///////////////////////////////////////////////////////////////////////////////////////////////////////////


img_cudaprocess::img_cudaprocess() {
//    std::cout<<"This is img_cudaprocess()!"<<std::endl;

}

img_cudaprocess::~img_cudaprocess() {
    release();
//    std::cout<<"This is ~img_process()!"<<std::endl;
}

void img_cudaprocess::release() {

    // 1.下采样变量
    cudaFree(img_gpu);
    cudaFree(imgSmall_gpu);

    // 2.灰度图像
    cudaFree(imggray_result);
    cudaFree(imggray_gpu);
    cudaFree(d_in);
    cudaFree(d_out);

    // 3.图像灰度等级
    cudaFree(imggraylevel_gpu_src);
    cudaFree(imggraylevel_gpu_result);

    // 4.均衡化
    cudaFree(imgequalization_gpu_src);
    cudaFree(imgequalization_gpu_result);

    // 5.图像素描
    cudaFree(imgSketch_gpu_src);
    cudaFree(imgSketch_gpu_result);

    // 6.sobel边缘检测
    cudaFree(imgsobel_src);
    cudaFree(imgsobel_result);
}

void img_cudaprocess::Init(int sc, int gray_level, Gpu_Mem gpu_mem) {

    cudaMem = gpu_mem;
    scale = sc;
    level = gray_level;

}

void img_cudaprocess::process(cv::Mat im, std::string im_path, std::string im_name, int frame1){

    image = im;
    if(image.empty()){
        std::cout<<"Could not open or find the image"<<std::endl;
        return;
    }

    frame_num = frame1;
    imgW = im.cols;
    imgH = im.rows;
    img_path = im_path;
    img_name = im_name;
}

void img_cudaprocess::ImgDownSampling(){
    clock_t start, finish;
    double duration;
    start = clock();

    const int imageSize = imgW * imgH * 3;

    if(img_gpu != NULL)
        cudaFree(img_gpu);
    if(imgSmall_gpu != NULL)
        cudaFree(imgSmall_gpu);

    // 图像下采样空间申请
    cudaMalloc((void**) &img_gpu, imageSize);
    cudaMalloc((void**) &imgSmall_gpu, imageSize / scale / scale);
    cudaMemcpy(img_gpu, image.data, imageSize, cudaMemcpyHostToDevice);

//    cudaMem.createGpuMemory((void**) &img_gpu, imageSize);
//    cudaMem.createGpuMemory((void**) &imgSmall_gpu, imageSize / scale /scale);
    if(img_gpu == NULL || imgSmall_gpu == NULL)
    {
        printf("create cuda memory is failed！！！！！！");
        std::exit(0);
    }
    cudaMemcpy(img_gpu, image.data, imageSize, cudaMemcpyHostToDevice);

//    std::cout<<"This is ImgDownSampling function!"<<std::endl;
    const int SW = imgW / scale;
    const int SH = imgH / scale;
    _resize_n8u<<<(SW * 3 * SH + THREAD_DIM_X - 1)/THREAD_DIM_X, THREAD_DIM_X>>>(img_gpu, imgW, imgH, 3, imgSmall_gpu, SW, SH);
    img_small.create(imgH / scale, imgW / scale, CV_8UC3);
    cudaMemcpy(img_small.data, imgSmall_gpu, SW * 3 * SH, cudaMemcpyDeviceToHost);
//    cudaMem.releaseGpuMemory(img_gpu);
//    cudaMem.releaseGpuMemory(imgSmall_gpu);
//    img_gpu = NULL;
//    imgSmall_gpu = NULL;
//    cv::imshow("11111", img_small);


    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("CUDA ImgDownSampling(): %f seconds\n", duration);
}

//void img_process::ImgBGR2GRAY(){
//
//    // 灰度操作
//    cudaMalloc((void**) &imggray_gpu, imgW * imgH * sizeof(uint8_t));
//    cudaMemcpy(imggray_gpu, image.data, imgW * imgH * sizeof(uint8_t), cudaMemcpyHostToDevice);
//    cudaMalloc((void**) &imggray_result, imgW * imgH * sizeof(uint8_t));
//
//    bgr2gray_kernel<<<(imgW * imgH * 3 + THREAD_DIM_X -1) / THREAD_DIM_X, THREAD_DIM_X>>>(imggray_gpu, imgW, imgH, imggray_result);
//    img_gray.create(imgH, imgW, CV_8UC3);
//    cudaMemcpy(img_gray.data, imggray_result, imgW * 3 * imgH, cudaMemcpyDeviceToHost);
//
//}

void img_cudaprocess::ImgRGB2GRAY(){

    clock_t start, finish;
    double duration;
    start = clock();

    if(d_in != NULL) {
        cudaFree(d_in);
        d_in = NULL;
    }
    if(d_out != NULL) {
        cudaFree(d_out);
        d_out = NULL;
    }

    cudaMalloc((void**)&d_in, imgW*imgH*sizeof(uchar3));
    cudaMalloc((void**)&d_out, imgW * imgH * sizeof(unsigned char));
    cudaMemcpy(d_in, image.data, imgW*imgH*sizeof(uchar3), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((imgW + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (imgH + threadsPerBlock.y - 1) / threadsPerBlock.y);
    _rgb2gray<<<blocksPerGrid, threadsPerBlock>>>(d_in, imgH, imgW, d_out);
    img_gray.create(imgH, imgW, CV_8UC1);
    cudaMemcpy(img_gray.data, d_out, imgH * imgW * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("CUDA ImgRGB2GRAY(): %f seconds\n", duration);
}

void img_cudaprocess::ImgGrayLevel(){

    clock_t start, finish;
    double duration;
    start = clock();

    if(imggraylevel_gpu_src != NULL)
        cudaFree(imggraylevel_gpu_src);
    if(imggraylevel_gpu_result != NULL)
        cudaFree(imggraylevel_gpu_result);

    if(img_gray.empty())
    {
        ImgRGB2GRAY();
    }

    // 调整图像灰度等级的gpu空间申请
//    cv::Mat im_gray = cv::imread(img_path, cv::IMREAD_GRAYSCALE); // 暂时使用opencv读取灰度图像后续需要改成cuda实现
    cudaMalloc((void**) &imggraylevel_gpu_src, imgW * imgH);
    cudaMalloc((void**) &imggraylevel_gpu_result, imgW * imgH);
    cudaMemcpy(imggraylevel_gpu_src, img_gray.data, imgW * imgH, cudaMemcpyHostToDevice);

    _gray_level<<<(imgW * imgH + THREAD_DIM_X - 1) / THREAD_DIM_X, THREAD_DIM_X>>>(imggraylevel_gpu_src, imgW, imgH, 1, imggraylevel_gpu_result, level);
    imggraylevel_mat.create(imgH, imgW, CV_8UC1);
    cudaMemcpy(imggraylevel_mat.data, imggraylevel_gpu_result, imgW * imgH, cudaMemcpyDeviceToHost);

    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("CUDA ImgGrayLevel(): %f seconds\n", duration);
}

void img_cudaprocess::GrayEqualization() {

    clock_t start, finish;
    double duration;
    start = clock();

    if (imgequalization_gpu_result != NULL){

        cudaFree(imgequalization_gpu_result);
        imgequalization_gpu_result = NULL;
    }

    for(int i = 0; i < 256; i++){
        nSumPix[i] = 0;
        nProDis[i] = 0.0;
        nSumProDis[i] = 0.0;
        EqualizeSumPix[i] = 0;
    }

//    cv::Mat imm = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    if(img_gray.empty())
    {
        ImgRGB2GRAY();
    }

    const int imageSize = imgH * imgW;


    // 统计图像的像素调用_cuSumPix核函数
    if(nSumPixGpu != NULL){
        cudaFree(nSumPixGpu);
        nSumPixGpu = NULL;
    }
    if(imgequalization_gpu_src != NULL) {
        cudaFree(imgequalization_gpu_src);
        imgequalization_gpu_src = NULL;
    }
//    std::cout<<"<<<<<<<<<<<<<<before<<<<<<<<<<<<<<<"<<std::endl;
//    for(int i = 0; i < 256; i++)
//    {
//        std::cout<<nSumPix[i]<<" ";
//        ssssss += nSumPix[i];
//    }
//    std::cout<<std::endl;

    cudaMalloc((void**)&nSumPixGpu, sizeof(int)*256);
    cudaMalloc((void**)&imgequalization_gpu_src, sizeof(uint8_t) * imgH * imgW);
    cudaMemcpy(imgequalization_gpu_src, img_gray.data, imgH * imgW * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(nSumPixGpu, nSumPix, sizeof(int)*256, cudaMemcpyHostToDevice);
    int thread_gray = 512;
    _cuSumPix<<<(imgH * imgW + thread_gray - 1) / thread_gray, thread_gray>>>(imgequalization_gpu_src, nSumPixGpu, imgW, imgH);
//    if(frame_num != 7)
//        _cuSumPix<<<(imgH * imgW + thread_gray - 1) / thread_gray, thread_gray>>>(imgequalization_gpu_src, nSumPixGpu, imgW, imgH);
//    else
//        _cuSumPix2<<<(imgH * imgW + thread_gray - 1) / thread_gray, thread_gray>>>(imgequalization_gpu_src, nSumPixGpu, imgW, imgH);
    cudaMemcpy(nSumPix, nSumPixGpu, sizeof(int)*256, cudaMemcpyDeviceToHost);

    ////////////////////////////////////
//    std::cout<<"<<<<<<<<<<<<<<after<<<<<<<<<<<<<<<"<<std::endl;
//    long long int ssssss = 0;
//    for(int i = 0; i < 256; i++)
//    {
//        std::cout<<nSumPix[i]<<" ";
//        ssssss += nSumPix[i];
//    }
//    std::cout<<std::endl;
//    std::cout<<ssssss<<std::endl;
//    std::cout<<imgH<<" "<<imgW<<std::endl;
//    if(ssssss != imgH * imgW) {
//        std::cout<<"pix num is not \" imgH * imgW \""<<std::endl;
//        std::exit(0);
//    }
    ////////////////////////////////////

//    std::cout<<"cuda像素值的概率："<<std::endl;
    // 每个像素值的概率计算
    for(int i = 0; i < 256; i++){
        nProDis[i] = (double)nSumPix[i] / imageSize;
//        std::cout<<nProDis[i]<<" ";
    }
//    std::cout<<std::endl;

    // 计算累计分布概率
//    std::cout<<"cuda像素值的累积概率："<<std::endl;
    nSumProDis[0] = nProDis[0];
    for(int i = 1; i < 256; i++){
        nSumProDis[i] = nSumProDis[i-1] + nProDis[i];
//        std::cout<<nSumProDis[i]<<" ";
    }
//    std::cout<<std::endl;

    // 计算均衡化之后像素值对应的灰度值
//    std::cout<<"cuda像素值均衡化之后的灰度值："<<std::endl;
    for(int i = 0; i < 256; i++)
    {
//        EqualizeSumPix[i] = cvRound((double)nSumProDis[i] * 255);
        EqualizeSumPix[i] = (int)(255 * nSumProDis[i] + 0.5);
//        std::cout<<EqualizeSumPix[i]<<" ";
    }
//    std::cout<<std::endl;

    if(mgpuEqualizedSumPix != NULL){
        cudaFree(mgpuEqualizedSumPix);
        mgpuEqualizedSumPix = NULL;
    }
    cudaMalloc((void**)&mgpuEqualizedSumPix, sizeof(int)*256);
    cudaMalloc((void**)&imgequalization_gpu_result, sizeof(uint8_t)*imageSize);

    // 获得均衡化之后的图片
//    if(imgequalization_gpu_src != NULL)
//        cudaFree(imgequalization_gpu_src);
    cudaMemcpy(mgpuEqualizedSumPix, EqualizeSumPix, 256 * sizeof(int), cudaMemcpyHostToDevice);
//    cudaMemcpy(imgequalization_gpu_src, img_gray.data, imgH * imgW * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(imgequalization_gpu_result, 0, sizeof(uint8_t)*imageSize, cudaMemcpyHostToDevice);
//    int TILE_WIDTH = 16;
//    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
//    dim3 dimGrid((imgW + TILE_WIDTH - 1) / TILE_WIDTH, (imgH + TILE_WIDTH - 1) / TILE_WIDTH);
//    _cugrayequalized<<<dimGrid, dimBlock>>>(imgequalization_gpu_src, imgequalization_gpu_result, mgpuEqualizedSumPix, imgW, imgH);
    _cugrayequalized<<<(imageSize + THREAD_DIM_X - 1) / THREAD_DIM_X, THREAD_DIM_X>>>(imgequalization_gpu_src, imgequalization_gpu_result, mgpuEqualizedSumPix, imageSize);
    imgequalization.create(imgH, imgW, CV_8UC1);
    cudaMemcpy(imgequalization.data, imgequalization_gpu_result, sizeof(uint8_t)*imageSize, cudaMemcpyDeviceToHost);

//    cv::imshow("sssss", imgequalization);
//    cv::waitKey(0);



    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("CUDA GrayEqualization(): %f seconds\n", duration);
}

void img_cudaprocess::RGBEqualization(){

    clock_t start, finish;
    double duration;
    start = clock();

    // https://blog.csdn.net/weixin_45875105/article/details/106807655
    cv::Mat channels[3];
    cv::Mat out_channels[3];
    cv::split(image, channels);

    for(int i = 0; i < 3; i++){
        int imageSize = imgH * imgW;
        for(int i = 0; i < 256; i++){
            nSumPix[i] = 0;
            nProDis[i] = 0.0;
            nSumProDis[i] = 0.0;
            EqualizeSumPix[i] = 0;
        }

        if(imgequalization_gpu_src != NULL){
            cudaFree(imgequalization_gpu_src);
            imgequalization_gpu_src = NULL;
        }

        if(imgequalization_gpu_result != NULL){
            cudaFree(imgequalization_gpu_result);
            imgequalization_gpu_result = NULL;
        }

        // 统计图像的像素调用_cuSumPix核函数
        cudaMalloc((void**)&nSumPixGpu, sizeof(int)*256);
        cudaMemcpy(nSumPixGpu, nSumPix, sizeof(int)*256, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&imgequalization_gpu_src, sizeof(uint8_t) * imgH * imgW);
        cudaMemcpy(imgequalization_gpu_src, channels[i].data, imgH * imgW, cudaMemcpyHostToDevice);
        _cuSumPix<<<(imgH * imgW + THREAD_DIM_X - 1) / THREAD_DIM_X, THREAD_DIM_X>>>(imgequalization_gpu_src, nSumPixGpu, imgW, imgH);
        cudaMemcpy(nSumPix, nSumPixGpu, sizeof(int)*256, cudaMemcpyDeviceToHost);

        // 每个像素值的概率计算
        for(int i = 0; i < 256; i++){
            nProDis[i] = (double)nSumPix[i] / imageSize;
        }

        // 计算累计分布概率
        for(int i = 0; i < 256; i++){
            nSumProDis[i] = nSumProDis[i-1] + nProDis[i];
        }

        // 计算均衡化之后像素值对应的灰度值
        for(int i = 0; i < 256; i++)
        {
            EqualizeSumPix[i] = cvRound((double)nSumProDis[i] * 255);
        }

        cudaMalloc((void**)&mgpuEqualizedSumPix, sizeof(int)*256);
        cudaMalloc((void**)&imgequalization_gpu_result, sizeof(uint8_t)*imageSize);

        // 获得均衡化之后的图片
        cudaMemcpy(mgpuEqualizedSumPix, EqualizeSumPix, 256  * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(imgequalization_gpu_src, channels[i].data, imgH * imgW, cudaMemcpyHostToDevice);
        cudaMemcpy(imgequalization_gpu_result, 0, sizeof(uint8_t)*imageSize, cudaMemcpyHostToDevice);
        _cugrayequalized<<<(imageSize + THREAD_DIM_X - 1) / THREAD_DIM_X, THREAD_DIM_X>>>(imgequalization_gpu_src, imgequalization_gpu_result, mgpuEqualizedSumPix, imageSize);
        out_channels[i].create(imgH, imgW, CV_8UC1);
        cudaMemcpy(out_channels[i].data, imgequalization_gpu_result, sizeof(uint8_t)*imageSize, cudaMemcpyDeviceToHost);

        // 释放空间
        cudaFree(nSumPixGpu);
        cudaFree(mgpuEqualizedSumPix);
    }

    cv::merge(out_channels, 3, imgRGBequalization);

    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("CUDA RGBEqualization(): %f seconds\n", duration);
}

// RGB变素描
// 1. 图像去色变成灰度图
// 2. 图像取反
// 3. 高斯滤波
// 4. 颜色减淡混合
void img_cudaprocess::RGBSketch(){

    clock_t start, finish;
    double duration;
    start = clock();

    if(imgSketch_gpu_src != NULL){
        cudaFree(imgSketch_gpu_src);
        imgSketch_gpu_src = NULL;
    }
    if(imgSketch_gpu_result != NULL){
        cudaFree(imgSketch_gpu_result);
        imgSketch_gpu_result = NULL;
    }

    // 1. 图像去色变成灰度图
    if(img_gray.empty())
    {
        ImgRGB2GRAY();
    }

    uint8_t* imgSketch_pixelnegate;  // 保存灰度图像取反的gpu指针

    int imagesize = imgW * imgH;
    cudaMalloc((void**)&imgSketch_gpu_src, sizeof(uint8_t)*imagesize);
    cudaMalloc((void**)&imgSketch_pixelnegate, sizeof(uint8_t)*imagesize);
    cudaMemcpy(imgSketch_gpu_src, img_gray.data, sizeof(uint8_t)*imagesize, cudaMemcpyHostToDevice);

    // 2. 图像取反
    // 调用核函数对图像进行取反操作
    _PixelNegate<<<(imagesize + THREAD_DIM_X - 1) / THREAD_DIM_X, THREAD_DIM_X>>>(imgSketch_gpu_src, imgSketch_pixelnegate, imagesize);

    // 3. 高斯滤波

//////////////////////////////****************************************///////////////////////////////////////////
    // 高斯卷积核参数创建
//    int size = 15;
//    double sigma = 50;
//    double temp_sum = 0;
//    int sum = 0;
//    double **GaussinBLur_kernal = new double *[size];
//    for(int i=0;i<size;i++)
//    {
//        GaussinBLur_kernal[i]=new double[size];  //动态生成矩阵
//    }
//    GetGaussianKernel(GaussinBLur_kernal, size, sigma, temp_sum);
//    sum = temp_sum;

//    for(int i = 0; i < size; i++)
//    {
//        for(int j = 0; j < size; j++)
//        {
//            std::cout<<GaussinBLur_kernal[i][j]<<"  ";
//        }
//        std::cout<<std::endl<<std::endl;
//    }

    // https://www.cnblogs.com/mtcnn/p/9411977.html
    // https://blog.csdn.net/tangbenjun/article/details/78018493
    // https://blog.csdn.net/qq_32563773/article/details/106274697?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-4.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-4.control
    // cuda实现高斯滤波

    // 把得到的高斯滤波核变成一维的
//    double gausArray[size*size];
//    double temp[size*size];
//    for(int i = 0; i < size * size; i++)
//    {
//        gausArray[i] = 0;
//        temp[i] = 0;
//    }
//    int array = 0;
//    for(int i = 0; i < size; i++)
//    {
//        for(int j = 0; j < size; j++)
//        {
//            gausArray[array] = GaussinBLur_kernal[i][j];
//            array++;
//        }
//    }
////
//    cudaMemcpyToSymbol(gausArray_gpu, gausArray, sizeof(double)*size*size);
//
//
//    uint8_t* imgSketch_pixelnegate_Gaussian;
//    cudaMalloc((void**)&imgSketch_pixelnegate_Gaussian, sizeof(uint8_t)*imagesize);
//    std::cout<<"1111111111111111111111"<<std::endl;
//    int blockNum = imgH;
//    int threadIdxNum = imgW;
//    _Gaussian<<<blockNum, threadIdxNum>>>(imgSketch_pixelnegate, imgSketch_pixelnegate_Gaussian, size, imagesize, imgW, imgH, sum);

//    int thead_xy = 32;
//    int bx = int(ceil((double)imgW / thead_xy));
//    int by = int(ceil((double)imgH / thead_xy));
//
//    dim3 grid(bx, by);
//    dim3 block(thead_xy, thead_xy);
//    gaussianFilter<<<grid,block>>>(imgSketch_pixelnegate, imgSketch_pixelnegate_Gaussian, imgW, imgH, sum);
//
//    std::cout<<"1111111111112312311111111111"<<std::endl;
//    cv::Mat img_GaussinBlur_src1;   // 图像取反操作的结果保存变量，也是高斯滤波的输入变量
//    img_GaussinBlur_src1.create(imgH, imgW, CV_8UC1);
//    cudaMemcpy(img_GaussinBlur_src1.data, imgSketch_pixelnegate_Gaussian, sizeof(uint8_t)*imagesize, cudaMemcpyDeviceToHost);
//    cv::imshow("gaussinblur", img_GaussinBlur_src1);
//    cv::waitKey(0);
//    std::exit(0);

//////////////////////////////****************************************///////////////////////////////////////////

    cv::Mat img_GaussinBlur_src;   // 图像取反操作的结果保存变量，也是高斯滤波的输入变量
    img_GaussinBlur_src.create(imgH, imgW, CV_8UC1);
    cudaMemcpy(img_GaussinBlur_src.data, imgSketch_pixelnegate, sizeof(uint8_t)*imagesize, cudaMemcpyDeviceToHost);


    cv::Mat img_GaussinBlur;  // 高斯滤波的结果保存变量
    cv::GaussianBlur(img_GaussinBlur_src, img_GaussinBlur, cv::Size(15,15), 50, 50);
//    cv::imshow("gaussinblur", img_GaussinBlur);
//    cv::waitKey(0);
//    std::exit(0);
    // 4. 颜色减淡混合
    uint8_t* img_GaussinBlur_gpu;  // 高斯滤波结果的gpu指针
    cudaMalloc((void**)&img_GaussinBlur_gpu, sizeof(uint8_t) * imagesize);
    cudaMemcpy(img_GaussinBlur_gpu, img_GaussinBlur.data, sizeof(uint8_t)*imagesize, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&imgSketch_gpu_result, sizeof(uint8_t) * imagesize);
    _ColorDodgeBlend<<<(imagesize + THREAD_DIM_X - 1) / THREAD_DIM_X, THREAD_DIM_X>>>(imgSketch_gpu_src, img_GaussinBlur_gpu, imgSketch_gpu_result, 255, imagesize);

    imgSketch_mat.create(imgH, imgW, CV_8UC1);
    cudaMemcpy(imgSketch_mat.data, imgSketch_gpu_result, sizeof(uint8_t)*imagesize, cudaMemcpyDeviceToHost);
//    cv::imshow("sssss", imgSketch_mat);
//    cv::waitKey();
//    std::exit(0);
    cudaFree(imgSketch_pixelnegate);
    cudaFree(img_GaussinBlur_gpu);
    cudaFree(gausArray_gpu);
//    cudaFree(imgSketch_pixelnegate_Gaussian);

    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("CUDA RGBSketch(): %f seconds\n", duration);
}

// https://blog.csdn.net/mgotze/article/details/77934138
void img_cudaprocess::ImgEdge_sobel() {

    //k
    if(img_gray.empty())
    {
        ImgRGB2GRAY();
    }

    clock_t start, finish;
    double duration;
    start = clock();

    if(imgsobel_src != NULL)
    {
        cudaFree(imgsobel_src);
        imgsobel_src = NULL;
    }

    if(imgsobel_result != NULL){
        cudaFree(imgsobel_result);
        imgsobel_result = NULL;
    }

    cudaMalloc((void**)&imgsobel_src, imgH * imgW * sizeof(uint8_t));
    cudaMalloc((void**)&imgsobel_result, imgH * imgW * sizeof(uint8_t));

    cv::Mat gaussImg;
    cv::GaussianBlur(img_gray, gaussImg, cv::Size(3,3), 0, 0);

    cudaMemcpy(imgsobel_src, gaussImg.data, imgH * imgW * sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((imgW + threadsPerBlock.x - 1) / threadsPerBlock.x, (imgH + threadsPerBlock.y - 1) / threadsPerBlock.y);
    sobelInCuda<<<blocksPerGrid, threadsPerBlock>>>(imgsobel_src, imgsobel_result, imgH, imgW);

    imgsobel_mat.create(imgH, imgW, CV_8UC1);
    cudaMemcpy(imgsobel_mat.data, imgsobel_result, imgH * imgW * sizeof(uint8_t), cudaMemcpyDeviceToHost);

//    cv::imshow("ssss", imgsobel_mat);
//    cv::waitKey();
//    std::exit(0);

    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("CUDA ImgEdge_sobel(): %f seconds\n", duration);

}

void img_cudaprocess::Binarization() {

    if(img_gray.empty())
    {
        ImgRGB2GRAY();
    }

    clock_t start, finish;
    double duration;
    start = clock();

    if(imgBinarization_result == NULL)
    {
        cudaFree(imgBinarization_result);
        imgBinarization_result = NULL;
    }

    int nSumPix[256];
    int* mGpuPix;

    for(int i = 0; i < 256; i++){
        nSumPix[i] = 0;
    }

    cudaMalloc((void**)&mGpuPix, sizeof(int) * 256);
    cudaMemcpy(mGpuPix, nSumPix, sizeof(int) * 256, cudaMemcpyHostToDevice);

    _cuSumPix<<<(imgW * imgH + THREAD_DIM_X - 1) / THREAD_DIM_X, THREAD_DIM_X>>>(d_out, mGpuPix, imgW, imgH);

    cudaMemcpy(nSumPix, mGpuPix, sizeof(int) * 256, cudaMemcpyDeviceToHost);

    long long int allPix = 0;
    int allNum = 0;
    int threshold = 0;

    for(int i = 0; i < 256; i++)
    {
        allPix += nSumPix[i] * i;
        allNum += nSumPix[i];
    }

    threshold = allPix / allNum;

    cudaMalloc((void**)&imgBinarization_result, sizeof(uint8_t) * imgH * imgW);
    _binarizationCuda<<<(imgW * imgH + THREAD_DIM_X - 1) / THREAD_DIM_X, THREAD_DIM_X>>>(d_out, imgBinarization_result, threshold, imgW, imgH);

    imgBinarization_mat.create(imgH, imgW, CV_8UC1);
    cudaMemcpy(imgBinarization_mat.data, imgBinarization_result, sizeof(uint8_t) * imgH * imgW, cudaMemcpyDeviceToHost);

    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("CUDA Binarization(): %f seconds\n", duration);
}