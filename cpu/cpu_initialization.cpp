#include "cpu_initialization.h"

using namespace cpu_space;

handle1::handle1(){

}

handle1::~handle1(){

}

void handle1::Init(){


    fun_process.Init(config.sc, config.gray_level);
}

void handle1::process(cv::Mat im, std::string im_path, std::string im_name){

    fun_process.process(im, im_path, im_name);

    if(config.down_flag){

    }
    if(config.gray_flag){
        fun_process.ImgRGB2GRAY();
        cv::imwrite(config.save_path + "/cpu/gray/" + fun_process.img_name, fun_process.img_gray);
    }
    if(config.graylevel_flag){
        fun_process.ImgGrayLevel();
        cv::imwrite(config.save_path + "/cpu/graylevel/" + fun_process.img_name, fun_process.imggraylevel_mat);
    }
    if(config.grayequalization_flag){
        fun_process.Equalization();
        cv::imwrite(config.save_path + "/cpu/grayequalization/" + fun_process.img_name, fun_process.imggrayequalization);
    }
    if(config.RGBequalization_flag){

    }
    if(config.RGBSketch_flag){

    }
    if(config.SobelEdge_flag){
    }
}