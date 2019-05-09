#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <vector>

#include "cuda_timer.h"
#include "cuda_utils.h"

#include "rroi.h"


#define PRINT(a) std::cout << #a << ": " << a << std::endl;

int main()
{
    const int batch_size = 2;
    const int channels = 1;
    const int height = 5;
    const int width = 5;
    const int pooled_height = 2;
    const int pooled_width = 2;
    const float spatial_scale = 1.0;


    // generate rois
    std::vector<std::vector<float>> rois {
        // batch_ind, xc, yc, w, h, angle
        {0, 1.5, 1.5, 2, 1, -45},
        {1, 2.5, 2.5, 2, 2, -90},
    };
    const int num_rois = rois.size();
    auto rois_size = num_rois * 6;

    std::vector<float> rois_flat(rois_size);
    for (int i = 0; i < num_rois; ++i)
    {
        const auto& r = rois[i];
        rois_flat[i*6] = r[0];
        rois_flat[i*6+1] = r[1];
        rois_flat[i*6+2] = r[2];
        rois_flat[i*6+3] = r[3];
        rois_flat[i*6+4] = r[4];
        rois_flat[i*6+5] = r[5];
    }

    // for (int i = 0; i < rois_size; ++i)
    // {
    //     std::cout << rois_flat[i] << " ";
    // }
    // std::cout << std::endl;

    // generate input
    const int bottom_data_size = batch_size * channels * height * width;
    std::vector<float> bottom_data(bottom_data_size );
    for (int i = 0; i < bottom_data_size; ++i)
    {
        bottom_data[i] = i;
    }

    // for (int i = 0; i < bottom_data_size; ++i)
    // {
    //     std::cout << bottom_data[i] << " ";
    // }
    // std::cout << std::endl;


    // host to device
    unique_ptr_device<float> bottom_data_d(nullptr);
    CUDA_CHECK(cudaMalloc((void **) &bottom_data_d, bottom_data_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(bottom_data_d.get(), &bottom_data[0], bottom_data_size * sizeof(float), cudaMemcpyHostToDevice));

    unique_ptr_device<float> rois_d(nullptr);
    CUDA_CHECK(cudaMalloc((void **) &rois_d, rois_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(rois_d.get(), &rois_flat[0], rois_size * sizeof(float), cudaMemcpyHostToDevice));

    unique_ptr_device<float> top_data_d(nullptr);
    const int top_data_size = num_rois * channels * pooled_height * pooled_width;
    CUDA_CHECK(cudaMalloc((void **) &top_data_d, top_data_size * sizeof(float)));


    // run kernel
    CUDATimer timer;
    timer.start();
    vincent_rroi_align(
        batch_size,
        num_rois,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        spatial_scale,
        bottom_data_d.get(),
        rois_d.get(),
        top_data_d.get(),
        0
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();
    std::cout << "vincent_rroi_align: " << timer.elapsed() << std::endl;

    
    PRINT(top_data_size)

    // copy to host
    std::vector<float> top_data_h(top_data_size);
    CUDA_CHECK(cudaMemcpy(&top_data_h[0], top_data_d.get(), top_data_size * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < top_data_size; ++i)
    {
        std::cout << top_data_h[i] << std::endl;
    }

    // BACKWARD PASS
    std::vector<float> top_diff(top_data_size, 1);

    unique_ptr_device<float> top_diff_d(nullptr);
    CUDA_CHECK(cudaMalloc((void **) &top_diff_d, top_data_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(top_diff_d.get(), &top_diff[0], top_data_size * sizeof(float), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemset(top_diff_d.get(), 1, top_data_size * sizeof(float)));  // set all to ones

    unique_ptr_device<float> bottom_diff_d(nullptr);
    CUDA_CHECK(cudaMalloc((void **) &bottom_diff_d, bottom_data_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(bottom_diff_d.get(), 0, bottom_data_size * sizeof(float)));  

    // run backward kernel
    timer.start();
    vincent_rroi_align_backward(
        batch_size,
        num_rois,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        spatial_scale,
        top_diff_d.get(),
        rois_d.get(),
        bottom_diff_d.get(),
        0
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();
    std::cout << "vincent_rroi_align: " << timer.elapsed() << std::endl;

    std::vector<float> bottom_diff(bottom_data_size);
    CUDA_CHECK(cudaMemcpy(&bottom_diff[0], bottom_diff_d.get(), bottom_data_size * sizeof(float), cudaMemcpyDeviceToHost));

    int ix = 0;
    for (int b = 0; b < batch_size; ++b)
    {
        for (int c = 0; c < channels; ++c)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    printf("%.2f, ", bottom_diff[ix++]); 
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
    return 0;
}
