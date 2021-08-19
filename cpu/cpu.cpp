#include <time.h>

#include "cpu.h"

#define CLOCKS_PER_SEC ((clock_t)1000)

void rgb2grayincpu(unsigned char * const d_in, unsigned char * const d_out,
                   uint imgheight, uint imgwidth)
{
    for(int i = 0; i < imgheight; i++)
    {
        for(int j = 0; j < imgwidth; j++)
        {
            d_out[i * imgwidth + j] = 0.299f * d_in[(i * imgwidth + j)*3]
                                      + 0.587f * d_in[(i * imgwidth + j)*3 + 1]
                                      + 0.114f * d_in[(i * imgwidth + j)*3 + 2];
        }
    }
}


unsigned  char get_value(int level, uchar v)
{
    int block_num = level - 1;
    int block_size = 256 / block_num;
    for (int i = 1; i <= block_num; i++)
    {
        if (v > block_size * i)
        {
            continue;
        }

        int mid_value = block_size * i / 2;
        int left = block_size * (i - 1);
        int right = block_size * i - 1;
        if (v < mid_value)
        {
            return left;
        }
        else
        {
            return right;
        }
    }

    return v;
}

void gray_level(unsigned  char * const d_in, unsigned  char * const d_out, int imgheight, int imgwidth, int level){
    for(int i = 0; i < imgheight; i++){
        for(int j = 0; j < imgwidth; j++){
            d_out[i * imgwidth + j] = get_value(level, d_in[i * imgwidth + j]);
        }
    }
}

// https://www.cnblogs.com/skyfsm/p/7767043.html
void grayequalized(cv::Mat& d_in, cv::Mat& d_out, int imgheight, int imgwidth){

    d_out = d_in.clone();

    int gray[256] = { 0 };  //记录每个灰度级别下的像素个数
    double gray_prob[256] = { 0 };  //记录灰度分布密度
    double gray_distribution[256] = { 0 };  //记录累计密度
    int gray_equal[256] = { 0 };  //均衡化后的灰度值

    int gray_sum = 0; // 像素总数

    gray_sum = imgwidth * imgheight;

    for(int i = 0; i < imgheight; i++){
        uchar* p = d_out.ptr<uchar>(i);
        for(int j = 0; j < imgwidth; j++){
            int value = p[j];
            gray[value]++;
        }
    }

    /////////////////////////////////////////////////////////////
//    std::cout<<"<<<<<<<<<<<<<<cpu<<<<<<<<<<<<<<<"<<std::endl;
//    int mm = 0;
//    for(int i = 0; i < 256; i++)
//    {
//        mm += gray[i];
//        std::cout<<gray[i]<<" ";
//    }
//    std::cout<<std::endl;
//    std::cout<<mm<<std::endl;
    /////////////////////////////////////////////////////////

//    std::cout<<"cpu像素值的概率："<<std::endl;
    // 统计灰度频率
    for(int i = 0; i < 256; i++){
        gray_prob[i] = ((double)gray[i] / gray_sum);
//        std::cout<<gray_prob[i]<<" ";
    }
//    std::cout<<std::endl;

    // 计算累计密度
//    std::cout<<"cpu像素值的累积概率："<<std::endl;
    gray_distribution[0] = gray_prob[0];
    for(int i = 1; i < 256; i++){
        gray_distribution[i] = gray_distribution[i-1] + gray_prob[i];
//        std::cout<<gray_distribution[i]<<" ";
    }
//    std::cout<<std::endl;

    // 重新计算均衡化后的灰度值，四舍五入。参考公式:(N - 1) * T + 0.5
//    std::cout<<"cpu像素值均衡化之后的灰度值："<<std::endl;
    for(int i = 0; i < 256; i++){
        gray_equal[i] = (uchar)(255 * gray_distribution[i] + 0.5);
//        std::cout<<gray_equal[i]<<" ";
    }
//    std::cout<<std::endl;

    // 直方图均衡化，更新原图每个点的像素值
    for(int i = 0; i < imgheight; i++){
        uchar* p = d_out.ptr<uchar>(i);
        for(int j = 0; j < imgwidth; j++){
            p[j] = gray_equal[p[j]];
        }
    }
}

img_cpuprocess::img_cpuprocess() {

}

img_cpuprocess::~img_cpuprocess() {

}

void img_cpuprocess::Init(int sc, int gray_level) {

    scale = sc;
    level = gray_level;
}

void img_cpuprocess::process(cv::Mat im, std::string im_path, std::string im_name){

    image = im;
    if(image.empty()){
        std::cout<<"Could not open or find the image"<<std::endl;
        return;
    }

    imgW = im.cols;
    imgH = im.rows;
    img_path = im_path;
    img_name = im_name;
}

void img_cpuprocess::ImgRGB2GRAY() {

    clock_t start, finish;
    double duration;
    start = clock();

    img_gray.create(imgH, imgW, CV_8UC1);
    rgb2grayincpu(image.data, img_gray.data, imgH, imgW);

    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("CPU ImgRGB2GRAY(): %f seconds\n", duration);
}

void img_cpuprocess::ImgGrayLevel(){

    clock_t start, finish;
    double duration;
    start = clock();

    imggraylevel_mat.create(imgH, imgW, CV_8UC1);
    if(img_gray.empty())
    {
        ImgRGB2GRAY();
    }
    gray_level(img_gray.data, imggraylevel_mat.data, imgH, imgW, level);

    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("CPU ImgRGB2GRAY(): %f seconds\n", duration);
}

void img_cpuprocess::Equalization(){

    clock_t start, finish;
    double duration;
    start = clock();

    imggrayequalization.create(imgH, imgW, CV_8UC1);
    if(img_gray.empty())
    {
        ImgRGB2GRAY();
    }

    grayequalized(img_gray, imggrayequalization, imgH, imgW);

    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("CPU GrayEqualization(): %f seconds\n", duration);
}