#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>


/* main */
int main(int argc, const char *argv[]) {
    //接受三个运行参数
    //1. 模型
    //2. 要预测的图片
    //3. label的文本
    if (argc < 4) {
        std::cerr << "usage: CppProject <path-to-exported-script-module> "
                  << "<path-to-image>  <path-to-category-text>\n";
        return -1;
    }

    //加载模型
    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);

    assert(module != nullptr);
    std::cout << "load model ok\n";

    //生成一个随机输入
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(torch::rand({64, 3, 224, 224}));

    // 计算网络一次前向传播的需要时间
    auto t = (double) cv::getTickCount();
    module->forward(inputs).toTensor();
    t = (double) cv::getTickCount() - t;
    printf("execution time = %gs\n", t / cv::getTickFrequency());
    inputs.pop_back();

    // 记载一张图片并且进行归一化
    cv::Mat image;
    image = cv::imread(argv[2], 1);
    cv::cvtColor(image, image, CV_BGR2RGB);  //转化为RGB三通道
    cv::Mat img_float;
    image.convertTo(img_float, CV_32F, 1.0 / 255);   //首先归一化到[0,1]区间
    cv::resize(img_float, img_float, cv::Size(224, 224));  //resize to 224，预训练的模型输入是batchsize x3 x 224 x 224
    //std::cout << img_float.at<cv::Vec3f>(56,34)[1] << std::endl;
    auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img_float.data, {1, 224, 224, 3});   //将cv::Mat转成tensor
    img_tensor = img_tensor.permute({0, 3, 1, 2});   //翻转让通道是第二个维度
    //均值归一化
    img_tensor[0][0] = img_tensor[0][0].sub_(0.485).div_(0.229);
    img_tensor[0][1] = img_tensor[0][1].sub_(0.456).div_(0.224);
    img_tensor[0][2] = img_tensor[0][2].sub_(0.406).div_(0.225);
    auto img_var = torch::autograd::make_variable(img_tensor, false);
    inputs.emplace_back(img_var);

    //对输入的图片进行前向传播计算
    torch::Tensor out_tensor = module->forward(inputs).toTensor();
    //std::cout << out_tensor.slice(/*dim=*/1, /*start=*/0, /*end=*/10) << '\n';

    // 加载label的文件
    std::string label_file = argv[3];
    std::ifstream rf(label_file.c_str());
    CHECK(rf) << "Unable to open labels file " << label_file;
    std::string line;
    std::vector<std::string> labels;
    while (std::getline(rf, line))
        labels.push_back(line);

    // 打印score是Top-5的预测label和score
    std::tuple<torch::Tensor, torch::Tensor> result = out_tensor.sort(-1, true);
    torch::Tensor top_scores = std::get<0>(result)[0];
    torch::Tensor top_idxs = std::get<1>(result)[0].toType(torch::kInt32);

    auto top_scores_a = top_scores.accessor<float, 1>();  //1是dim
    auto top_idxs_a = top_idxs.accessor<int, 1>();

    for (int i = 0; i < 5; ++i) {
        std::cout << "score: " << top_scores_a[i];
        std::cout << "  label: " << labels[top_idxs_a[i]] << std::endl;
    }

    return 0;
}