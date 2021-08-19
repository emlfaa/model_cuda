

#include "../utils_cuda/config.h"

#include "cpu.h"

namespace cpu_space{
    class handle1{

    public:
        handle1();
        ~handle1();

        void Init();

        void process(cv::Mat im, std::string im_path, std::string im_name);

    public:
        img_cpuprocess fun_process;
        Config config;
    };
}

