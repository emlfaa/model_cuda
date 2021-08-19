#include "cudamn.h"
//#ifndef _CONFIG_H_
#include "../utils_cuda/config.h"
//#define _CONFIG_H_

namespace cuda_space{
    class handle1{

    public:
        handle1();
        ~handle1();

        void Init(Gpu_Mem cudaMem);

        void process(cv::Mat im, std::string im_path, std::string im_name);

        void process_video(cv::Mat im, std::string im_path, std::string im_name, int frame_num);


    public:
        img_cudaprocess fun_process;
        Config config;
        Gpu_Mem gpu_mem;
    };
}

//#endif