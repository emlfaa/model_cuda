//
// Created by buyun on 2020/11/17.
//
#include <iostream>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <vector_functions.h>
#ifndef LEARN_CUDA_CUDA_H
#define LEARN_CUDA_CUDA_H

class GpuMem_message{
public:
    GpuMem_message();
    bool create(size_t _size);
    void destroy();
public:
    uint8_t *cuda_addr = NULL;
    size_t cuda_size = 0;
    bool usable = false;
    bool ever_used = false;
};

class Gpu_Mem{
public:
    Gpu_Mem();

    bool init_Gpu_Mem(const int width, const int height, const int sc);
    bool destroy_Gpu_Mem();
    bool createGpuMemory(void** memobj, size_t _size);
    bool releaseGpuMemory(uint8_t *memobj);

public:
    int imgW;
    int imgH;
    std::vector<GpuMem_message> gpu_mems;
};

class img_cudaprocess {
public:
    img_cudaprocess();
    ~img_cudaprocess();
    void release();

    void Init(int sc, int gray_level, Gpu_Mem gpu_mem);

    void process(cv::Mat im, std::string img_path, std::string im_name, int frame_num1);



    // 1.图像下采样
    void ImgDownSampling(); // 图像下采样

    // 2.图像灰度
    void ImgBGR2GRAY();   // 图像灰度方法一
    void ImgRGB2GRAY();   // 图像灰度方法二

    // 3.图像灰度等级调整
    // https://www.cnblogs.com/skyfsm/p/7586836.html
    void ImgGrayLevel(); // 图像灰度等级的调整

    // 4.均衡化
    void Equalization(cv::Mat in_equalization, cv::Mat out_equalization);  // 均衡化
    void GrayEqualization();  // 图像均衡化
    void RGBEqualization();   // RGB图像均衡化

    // 5.图像素描
    void RGBSketch();    // 图像素描
    void GetGaussianKernel(double **gaus, int size, double sigma, double& temp);

    // 6.sobel边缘检测
    void ImgEdge_sobel();

    // 7.二值化
    void Binarization();


public:

    Gpu_Mem cudaMem;

    int frame_num;
    int imgW;
    int imgH;
    int scale;  // 下采样倍数
    int level;  // 图像灰度等级
    cv::Mat image;
    std::string img_path;
    std::string img_name;

    // 1.下采样变量
    uint8_t *img_gpu;
    uint8_t *imgSmall_gpu;
    cv::Mat img_small;


    // 2.灰度图像操作的指针
    // 方法一
    uint8_t *imggray_result;
    uint8_t *imggray_gpu;
    // 方法二
    uchar3 *d_in;
    unsigned char *d_out;
    cv::Mat img_gray;

    // 3.调整图像灰度等级
    uint8_t *imggraylevel_gpu_src;  // 调整图像灰度等级的gpu指针
    uint8_t *imggraylevel_gpu_result;  // 调整图像灰度等级后结果的gpu指针
    cv::Mat imggraylevel_mat;

    // 4.灰度图像均衡化
    uint8_t *imgequalization_gpu_src;      // 灰度图像均衡化gpu指针
    uint8_t *imgequalization_gpu_result;   // 灰度图像均衡化结果gpu指针
    cv::Mat imgequalization;
    // RGB图像均衡化
    cv::Mat imgRGBequalization;
    // 统计像素的变量
    int* nSumPixGpu;        // 统计每个像素值的个数GPU
    int* mgpuEqualizedSumPix;
    int nSumPix[256] = {0};        // 统计每个像素值的个数CPU
    double nProDis[256] = {0};     // 计算每个灰度级占图像中的概率分布
    double nSumProDis[256] = {0};  // 计算累计分布概率
    int EqualizeSumPix[256] = {0}; //


    // 5.RGB图像素描
    uint8_t *imgSketch_gpu_src;
    uint8_t *imgSketch_gpu_result;
    cv::Mat imgSketch_mat;

    // 6.sobel边缘检测
    uint8_t *imgsobel_src;
    uint8_t *imgsobel_result;
    cv::Mat imgsobel_mat;

    // 7.二值化
    uint8_t  *imgBinarization_result;
    cv::Mat imgBinarization_mat;
};




#endif //LEARN_CUDA_CUDA_H
