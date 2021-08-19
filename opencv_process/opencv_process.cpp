#include <iostream>


#include "opencv_process.h"


img_opencvprocess::img_opencvprocess() {

}

img_opencvprocess::~img_opencvprocess() {

}

void img_opencvprocess::Init(int sc, int gray_level){



    scale = sc;
    level = gray_level;


};

void img_opencvprocess::process(cv::Mat im, std::string im_path, std::string im_name){

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

void img_opencvprocess::ImgRGB2GRAY() {
    cv::cvtColor(image, img_gray, CV_BGR2GRAY);
}

void img_opencvprocess::Equalization() {

    if(img_gray.empty()){
        ImgRGB2GRAY();
    }

    cv::equalizeHist(img_gray, imggrayequalization);
}

void img_opencvprocess::ImgEdge_sobel() {

    if(img_gray.empty()){
        ImgRGB2GRAY();
    }

    cv::Mat gray_gaussian, grad_x, grad_y;
    cv::GaussianBlur(img_gray,gray_gaussian, cv::Size(15, 15), 50, 50);
    cv::Sobel(gray_gaussian, grad_x, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::Sobel(gray_gaussian, grad_y, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

    cv::convertScaleAbs(grad_x, grad_x);
    cv::convertScaleAbs(grad_y, grad_y);

    cv::add(grad_x, grad_y, imgsobel_mat, cv::Mat(), CV_16S);
    cv::convertScaleAbs(imgsobel_mat, imgsobel_mat);

    cv::imshow("sobel", imgsobel_mat);
}