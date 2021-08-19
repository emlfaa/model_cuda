#ifndef CONFIG_H_
#define CONFIG_H_

#include <iostream>

class Config{
public:

    int data_type = 1;   // 0:图片  1:视频

    // image input
    std::string file_path = "../img/test";  // 原始数据目录
    std::string save_path = "../img/result";   // 预处理数据保存目录

    // video input

    bool video_flag = 1;    //
//    std::string video_path = "../img/768x576.avi";
    std::string video_path = "/home/ub/workspace/数据/视频 数据/原视频/201810181739529687.mp4";

    int sc = 8;
    int gray_level = 4;

    bool all_flag = true;      // 所有平台下的预处理输出
    bool cpu_flag = true;      // cpu下的输出
    bool cuda_flag = true;     // cuda下的输出
    bool opencv_flag = true;   // opencv下的输出


    bool down_flag = true;                 // 下采样
    bool gray_flag = true;                 // 灰度化
    bool graylevel_flag = true;            // 灰度等级调整
    bool grayequalization_flag = true;     // 灰度图均衡化
    bool RGBequalization_flag = true;      // 彩色图像均衡化
    bool RGBSketch_flag = true;            // 彩色图像变素描
    bool SobelEdge_flag = true;            // sobel算子边缘检测
    bool Binarization_flag = true;         // 二值化


};

#endif