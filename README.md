# Torch Demo

Pytorch LibTorch demo

# setup
1. Download C++ LibTorch library at https://pytorch.org/get-started/locally/
2. Download and build C++ library OpenCV at https://opencv.org/releases/
3. Setup your libs path on CMakeList.txt
```
set(Torch_DIR ~/your/path/here/libtorch/share/cmake/Torch)
set(OpenCV_DIR ~/your/path/here/opencv-4.7.0/build)
```

# Hints
For libtroch "Pre-cxx11 ABI" can cause compiler conflicts, instead use cxx11 ABI

# Step
1. python3 tracing.py
2. build the c++ program
3. ./TorchDemo ../model.pt ../dog.png ../synset_words.txt


# more informattion
https://jackan.cn/2018/12/23/libtorch-test/#more
