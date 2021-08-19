#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

class img_opencvprocess{
public:
    img_opencvprocess();
    ~img_opencvprocess();

    void Init(int sc, int gray_level);

    void process(cv::Mat im, std::string img, std::string im_name);

    void ImgRGB2GRAY();

    // 4.均衡化
    void Equalization();

    // 6.sobel边缘检测
    void ImgEdge_sobel();

public:
    cv::Mat image;
    int imgW;
    int imgH;
    int scale;
    int level;
    std::string img_path;
    std::string img_name;

    // 2.灰度图像操作
    cv::Mat img_gray;

    // 4.灰度图像均衡化
    cv::Mat imggrayequalization;

    // 6.sobel边缘检测
    cv::Mat imgsobel_mat;

};
