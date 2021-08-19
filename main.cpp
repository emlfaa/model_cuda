#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <unistd.h>
#include <chrono>
#include <thread>
#include <time.h>
#include "utils_cuda/file_process.h"

#include "opencv_process/opencv_initialization.h"
#include "cpu/cpu_initialization.h"
#include "cuda/cuda_initialization.h"
#include "../utils_cuda/config.h"
#include "../utils_cuda/memory_pool.h"
#include "../tensorrt/yolov5/yolov5.h"

#define CLOCKS_PER_SEC ((clock_t)1000000)
//#define CLOCKS_PER_SEC ((clock_t)1000)
using namespace std;
using namespace cv;


int main() {
    
    Config config;

    if(config.data_type == 0)
    {

        file_process file_process;

        vector<string> type_file;
        if(config.all_flag)
        {
            type_file.push_back("opencv");
            type_file.push_back("cpu");
            type_file.push_back("cuda");
            file_process.create_file(config.save_path, type_file);
        } else
        {
            if(config.cpu_flag)
                type_file.push_back("cpu");
            if(config.cuda_flag)
                type_file.push_back("cuda");
            if(config.opencv_flag)
                type_file.push_back("opencv");
            file_process.create_file(config.save_path, type_file);
        }

        vector<string> fun_file;
        if(config.down_flag)
            fun_file.push_back("down");
        if(config.gray_flag)
            fun_file.push_back("gray");
        if(config.graylevel_flag)
            fun_file.push_back("graylevel");
        if(config.grayequalization_flag)
            fun_file.push_back("grayequalization");
        if(config.RGBequalization_flag)
            fun_file.push_back("RGBequalization");
        if(config.RGBSketch_flag)
            fun_file.push_back("RGBSketch");
        if(config.SobelEdge_flag)
            fun_file.push_back("SobelEdge");
        if(config.Binarization_flag)
            fun_file.push_back("Binarization");
        for(int i = 0; i < type_file.size(); i++){
            file_process.create_file(config.save_path + "/" + type_file[i], fun_file);
        }

        opencv_space::handle1 opencv_handle;
        opencv_handle.Init();
        cpu_space::handle1 cpu_handle;
        cpu_handle.Init();

        vector<string> first_filenames;
        file_process.GetFileNames(config.file_path,first_filenames);
        for(int i = 0; i < first_filenames.size(); i++) {

            int position = first_filenames[i].find_last_of("/"); //找到最后一个/的位置
            string img_file = first_filenames[i].substr(position + 1, first_filenames[i].size() - position); // 图像文件地址

            // 读取每一个图像文件地址下的所有图像
            vector<string> img_path;
            file_process.GetFileNames(first_filenames[i], img_path);

            // 循环处理每一张图片
            for (int j = 0; j < img_path.size(); j++) {
                int position_last = img_path[j].find_last_of("/");  //获取图片名称前面/的位置

                string img_name = img_path[j].substr(position_last + 1, img_path[j].size() - position_last); // 获取图片的名字

                std::cout << img_path[j] << std::endl;
                cv::Mat image = cv::imread(img_path[j]);
                if (image.empty()) {
                    std::cout << "Could not open or find the image" << std::endl;
                    return -1;
                }

                Gpu_Mem cudaMem;
                cudaMem.init_Gpu_Mem(image.cols, image.rows, config.sc);
                cuda_space::handle1 cuda_handle;
                cuda_handle.Init(cudaMem);
                if (config.opencv_flag)
                    opencv_handle.process(image, img_path[j], img_name);
                if (config.cpu_flag)
                    cpu_handle.process(image, img_path[j], img_name);
                if (config.cuda_flag)
                    cuda_handle.process(image, img_path[j], img_name);

            }

        }


    }
    else if(config.data_type == 1)
    {
        cv::VideoCapture capture;
        if(config.video_flag){
            capture.open(config.video_path);
        } else
        {
          capture.open(0);
        }

        yolov5 yolov5_inference;
        yolov5_inference.Init();

        cv::Mat frame;
        capture>>frame;
        if(frame.empty())
        {
            printf("There is not frame!!!!!!!!!");
            std::exit(0);
        }
        Gpu_Mem cudaMem;
        cudaMem.init_Gpu_Mem(frame.cols, frame.rows, config.sc);
        cuda_space::handle1 cuda_handle;
        cuda_handle.Init(cudaMem);

        int frame_num = 1;
        clock_t start, finish_cuda, finsh_yolov5;
        double duration_cuda, duration_yolov5, duration_all;
        while(!frame.empty()){
            printf("----------------time Statistics----------------\n");
            std::cout<<"frame num: "<<frame_num<<std::endl;
            start = clock();
//            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            cuda_handle.process_video(frame, "", "", frame_num);
            finish_cuda = clock();
            yolov5_inference.Inference(frame);
            finsh_yolov5 = clock();

            duration_cuda = (double)(finish_cuda - start) / CLOCKS_PER_SEC;
            duration_yolov5 = (double)(finsh_yolov5 - finish_cuda) / CLOCKS_PER_SEC;
            duration_all = (double)(finsh_yolov5 - start) / CLOCKS_PER_SEC;

            printf("CUDA time: %f seconds\n", duration_cuda);
            printf("YOLOV5 time: %f seconds\n", duration_yolov5);
            printf("ONE FRAME process time: %f seconds\n", duration_all);
//            printf("----------------time Statistics----------------\n");
            cv::waitKey( 1000);

            capture>>frame;
            frame_num++;

        }
    }

    return 0;
}