#include "opencv_process.h"
#include "../utils_cuda/config.h"

namespace opencv_space{
    class handle1{

    public:
        handle1();
        ~handle1();

        void Init();

        void process(cv::Mat im, std::string im_path, std::string im_name);



    public:
        img_opencvprocess fun_process;
        Config config;
    };
}