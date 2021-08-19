#include "cuda_initialization.h"
#include "cudamn.h"
using namespace cuda_space;

handle1::handle1(){

}

handle1::~handle1(){

}

void handle1::Init(Gpu_Mem gpu_mem){

    fun_process.Init(config.sc, config.gray_level, gpu_mem);

    cv::namedWindow("src", CV_WINDOW_NORMAL);
    cv::namedWindow("down");
    cv::namedWindow("gray", CV_WINDOW_NORMAL);
    cv::namedWindow("graylevel", CV_WINDOW_NORMAL);
    cv::namedWindow("grayequalization", CV_WINDOW_NORMAL);
    cv::namedWindow("RGBequalization", CV_WINDOW_NORMAL);
    cv::namedWindow("RGBSketch", CV_WINDOW_NORMAL);
    cv::namedWindow("SobelEdge", CV_WINDOW_NORMAL);
    cv::namedWindow("Binarization", CV_WINDOW_NORMAL);

    cv::moveWindow("src", 0, 0);
    cv::moveWindow("down", 0, 300);
    cv::moveWindow("gray", 0, 580);
    cv::moveWindow("graylevel", 400, 0);
    cv::moveWindow("grayequalization", 400, 300);
    cv::moveWindow("RGBequalization", 400, 580);
    cv::moveWindow("RGBSketch", 730, 0);
    cv::moveWindow("SobelEdge", 730, 300);
    cv::moveWindow("Binarization", 0, 840);
}

void handle1::process(cv::Mat im, std::string im_path, std::string im_name){

    fun_process.process(im, im_path, im_name, 0);

    if(config.down_flag){
        fun_process.ImgDownSampling();
        cv::imwrite(config.save_path + "/cuda/down/" + fun_process.img_name, fun_process.img_small);
    }
    if(config.gray_flag){
        fun_process.ImgRGB2GRAY();
        cv::imwrite(config.save_path + "/cuda/gray/" + fun_process.img_name, fun_process.img_gray);
    }
    if(config.graylevel_flag){
        fun_process.ImgGrayLevel();
        cv::imwrite(config.save_path + "/cuda/graylevel/" + fun_process.img_name, fun_process.imggraylevel_mat);
    }
    if(config.grayequalization_flag){
        fun_process.GrayEqualization();
        cv::imwrite(config.save_path + "/cuda/grayequalization/" + fun_process.img_name, fun_process.imgequalization);
    }
    if(config.RGBequalization_flag){
        fun_process.RGBEqualization();
        cv::imwrite(config.save_path + "/cuda/RGBequalization/" + fun_process.img_name, fun_process.imgRGBequalization);
    }
    if(config.RGBSketch_flag){
        fun_process.RGBSketch();
        cv::imwrite(config.save_path + "/cuda/RGBSketch/" + fun_process.img_name, fun_process.imgSketch_mat);
    }
    if(config.SobelEdge_flag){
        fun_process.ImgEdge_sobel();
        cv::imwrite(config.save_path + "/cuda/SobelEdge/" + fun_process.img_name, fun_process.imgsobel_mat);
    }
    if(config.Binarization_flag){
        fun_process.Binarization();
        cv::imwrite(config.save_path + "/cuda/Binarization/" + fun_process.img_name, fun_process.imgBinarization_mat);
    }

}

void handle1::process_video(cv::Mat im, std::string im_path, std::string im_name, int frame_num){

    fun_process.process(im, im_path, im_name, frame_num);

    if(config.down_flag){
        fun_process.ImgDownSampling();
    }
    if(config.gray_flag){
        fun_process.ImgRGB2GRAY();

    }
    if(config.graylevel_flag){
        fun_process.ImgGrayLevel();

    }
    if(config.grayequalization_flag){
        fun_process.GrayEqualization();

    }
    if(config.RGBequalization_flag){
        fun_process.RGBEqualization();

    }
    if(config.RGBSketch_flag){
        fun_process.RGBSketch();

    }
    if(config.SobelEdge_flag){
        fun_process.ImgEdge_sobel();

    }
    if(config.Binarization_flag){
        fun_process.Binarization();
    }


    cv::imshow("src", fun_process.image);
    cv::imshow("down", fun_process.img_small);
    cv::imshow("gray", fun_process.img_gray);
    cv::imshow("graylevel", fun_process.imggraylevel_mat);
    cv::imshow("grayequalization", fun_process.imgequalization);
    cv::imshow("RGBequalization", fun_process.imgRGBequalization);
    cv::imshow("RGBSketch", fun_process.imgSketch_mat);
    cv::imshow("SobelEdge", fun_process.imgsobel_mat);
    cv::imshow("Binarization", fun_process.imgBinarization_mat);
//    cv::waitKey(10);
}