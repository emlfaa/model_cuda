#include <iostream>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <vector_functions.h>

class img_cpuprocess{
public:
    img_cpuprocess();
    ~img_cpuprocess();

    void Init(int sc, int gray_level);

    void process(cv::Mat im, std::string img_path, std::string im_name);

    void ImgRGB2GRAY();

    void ImgGrayLevel();

    void Equalization();

public:
    int imgW;
    int imgH;
    int scale;
    int level;
    cv::Mat image;
    std::string img_path;
    std::string img_name;

    // 2.灰度图像操作的指针
    cv::Mat img_gray;

    // 3.调整图像灰度等级
    cv::Mat imggraylevel_mat;

    // 4.灰度图像均衡化
    cv::Mat imggrayequalization;
};
