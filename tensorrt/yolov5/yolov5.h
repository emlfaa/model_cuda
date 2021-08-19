#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <vector>
#include <string>

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "NvInferRuntime.h"
#include "NvInferRuntimeCommon.h"
#include "cuda_runtime_api.h"

class yolov5{
public:

    yolov5();
    ~yolov5();

    void Init();
    void Inference(cv::Mat video_frame);

    void test_Inference(cv::Mat video_frame);

public:

    cv::Mat show_frame;

    char *trtModelStream{ nullptr };
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;

    std::string engine_name = "/home/ub/workspace/codespace/cloth/learn_cuda/tensorrt/yolov5/yolov5s.engine";
};