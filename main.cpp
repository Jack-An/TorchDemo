#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>


/* main */
int main(int argc, const char *argv[]) {
    //Accept three running parameters
     //1. model
     //2. The picture to be predicted
     //3. The text of the label
    if (argc < 4) {
        std::cerr << "usage: CppProject <path-to-exported-script-module> "
                  << "<path-to-image>  <path-to-category-text>\n";
        return -1;
    }

    //load model
    torch::jit::script::Module module = torch::jit::load(argv[1]);

    assert(&module != nullptr);
    std::cout << "load model ok\n";

    //generate a random input
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(torch::rand({64, 3, 224, 224}));

    // Calculate the time required for a forward pass of the network
    auto t = (double) cv::getTickCount();
    module.forward(inputs).toTensor();
    t = (double) cv::getTickCount() - t;
    printf("execution time = %gs\n", t / cv::getTickFrequency());
    inputs.pop_back();

    // Record a picture and normalize it
    cv::Mat image;
    image = cv::imread(argv[2], 1);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);  //Convert to RGB three-channel
    cv::Mat img_float;
    image.convertTo(img_float, CV_32F, 1.0 / 255);   //First normalize to [0,1] interval
    cv::resize(img_float, img_float, cv::Size(224, 224));  //resize to 224, the pre-trained model input is batchsize x3 x 224 x 224
    //std::cout << img_float.at<cv::Vec3f>(56,34)[1] << std::endl;
    auto img_tensor = torch::from_blob(img_float.data, {1, 224, 224, 3}).to(torch::kCPU); //Convert cv::Mat to tensor
    img_tensor = img_tensor.permute({0, 3, 1, 2});   //Flip so channel is the second dimension
    //mean normalization
    img_tensor[0][0] = img_tensor[0][0].sub_(0.485).div_(0.229);
    img_tensor[0][1] = img_tensor[0][1].sub_(0.456).div_(0.224);
    img_tensor[0][2] = img_tensor[0][2].sub_(0.406).div_(0.225);
    auto img_var = torch::autograd::make_variable(img_tensor, false);
    inputs.emplace_back(img_var);

    //Perform forward propagation calculations on the input image
    torch::Tensor out_tensor = module.forward(inputs).toTensor();
    //std::cout << out_tensor.slice(/*dim=*/1, /*start=*/0, /*end=*/10) << '\n';

    // Load the label file
    std::string label_file = argv[3];
    std::ifstream rf(label_file.c_str());
    CHECK(rf) << "Unable to open labels file " << label_file;
    std::string line;
    std::vector<std::string> labels;
    while (std::getline(rf, line))
        labels.push_back(line);

    // The print score is the predicted label and score of Top-5
    std::tuple<torch::Tensor, torch::Tensor> result = out_tensor.sort(-1, true);
    torch::Tensor top_scores = std::get<0>(result)[0];
    torch::Tensor top_idxs = std::get<1>(result)[0].toType(torch::kInt32);

    auto top_scores_a = top_scores.accessor<float, 1>();  //1 is dim
    auto top_idxs_a = top_idxs.accessor<int, 1>();

    for (int i = 0; i < 5; ++i) {
        std::cout << "score: " << top_scores_a[i];
        std::cout << "  label: " << labels[top_idxs_a[i]] << std::endl;
    }

    return 0;
}
