
#include "opencv_initialization.h"

using namespace opencv_space;

handle1::handle1(){

}

handle1::~handle1(){

}

void handle1::Init(){

    fun_process.Init(config.sc, config.gray_level);
}

void handle1::process(cv::Mat im, std::string im_path, std::string im_name) {

    fun_process.process(im, im_path, im_name);

    if(config.down_flag){

    }
    if(config.gray_flag){
        fun_process.ImgRGB2GRAY();
        cv::imwrite(config.save_path + "/opencv/gray/" + fun_process.img_name, fun_process.img_gray);
    }

    if(config.graylevel_flag){

    }
    if(config.grayequalization_flag){
        fun_process.Equalization();
        cv::imwrite(config.save_path + "/opencv/grayequalization/" + fun_process.img_name, fun_process.imggrayequalization);
    }
    if(config.RGBequalization_flag){

    }
    if(config.RGBSketch_flag){

    }
    if(config.SobelEdge_flag){
        fun_process.ImgEdge_sobel();
        cv::imwrite(config.save_path + "/opencv/SobelEdge/" + fun_process.img_name, fun_process.imgsobel_mat);
    }
}