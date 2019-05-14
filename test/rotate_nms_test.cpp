#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <memory>

#include "cuda_timer.h"
#include "cuda_utils.h"

#include "rotate_nms.h"

int main()
{
    const float nms_thresh = 0.5;
    const int max_output = -1;
    const int height = 100;
    const int width = 100;

    const int num_rois = 2048;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<float> rois_flat(num_rois * 5);
    for (auto i = 0; i < num_rois; i++) {
        rois_flat[i*5+0] = width / 2;
        rois_flat[i*5+1] = height / 2;
        std::uniform_real_distribution<float> dist_w(width / 10, width / 1.5);
        std::uniform_real_distribution<float> dist_h(height / 10, height / 1.5);
        std::uniform_real_distribution<float> dist_angle(-90, 90);
        rois_flat[i*5+2] = dist_w(gen);
        rois_flat[i*5+3] = dist_h(gen);
        rois_flat[i*5+4] = dist_angle(gen);
    }

    // // generate rois
    // std::vector<std::vector<float>> rois {
    //     // xc, yc, w, h, angle
    //     // {1.5, 1.5, 2, 1, -45},
    //     {2.5, 2.5, 3, 2, -90},
    //     {2.5, 2.5, 3, 2.1, -90},
    //     {2.5, 2.5, 3, 23, -45},
    // };
    // const int num_rois = rois.size();

    // std::vector<float> rois_flat(rois_size);
    // for (int i = 0; i < num_rois; ++i)
    // {
    //     const auto& r = rois[i];
    //     rois_flat[i*5] = r[0];
    //     rois_flat[i*5+1] = r[1];
    //     rois_flat[i*5+2] = r[2];
    //     rois_flat[i*5+3] = r[3];
    //     rois_flat[i*5+4] = r[4];
    // }

    auto rois_size = num_rois * 5;

    unique_ptr_device<float> rois_d(nullptr);
    CUDA_CHECK(cudaMalloc((void **) &rois_d, rois_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(rois_d.get(), &rois_flat[0], rois_size * sizeof(float), cudaMemcpyHostToDevice));

    unique_ptr_device<int64_t> out_keep(nullptr);

    CUDATimer timer;
    timer.start();
    int num_to_keep = rotate_nms_cuda(
        rois_d.get(),
        out_keep.get(),
        num_rois, 
        nms_thresh, 
        max_output
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();

    std::cout << "rotate_nms: " << timer.elapsed() << std::endl;
    printf("num_to_keep: %d\n", num_to_keep);

    return 0;   
}
