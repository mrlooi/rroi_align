#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <random>

void gen_testcase(const std::string& filename)
{
  const int batch_size = 2;
  const int num_rois = 100;
  const int channels = 1000;
  const int height = 100;
  const int width = 100;
  const int pooled_height = 10;
  const int pooled_width = 10;
  const float spatial_scale = 1.0;

  std::random_device rd;
  std::mt19937 gen(rd());

  auto bottom_data_size = batch_size * channels * height * width;
  std::unique_ptr<float []> bottom_data(new float[bottom_data_size]);
  for (auto i = 0; i < bottom_data_size; i++) {
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    bottom_data[i] = dist(gen);
  }

  std::unique_ptr<float []> rois(new float[num_rois * 6]);
  for (auto i = 0; i < num_rois; i++) {
    std::uniform_int_distribution<int> dist_batch_id(0, batch_size-1);
    rois[i*6] = dist_batch_id(gen);
    rois[i*6+1] = width / 2;
    rois[i*6+2] = height / 2;
    rois[i*6+3] = width / 4;
    rois[i*6+4] = height / 4;
    std::uniform_real_distribution<float> dist_angle(-90, 90);
    rois[i*6+5] = dist_angle(gen);
  }

  std::fstream fout(filename, std::ios::out);
  fout << batch_size      << " "
       << num_rois        << " "
       << channels        << " "
       << height          << " "
       << width           << " "
       << pooled_height   << " "
       << pooled_width    << " "
       << spatial_scale   << std::endl;

  for (auto i = 0; i < bottom_data_size; i++) {
    fout << bottom_data[i] << " ";
  }
  fout << std::endl;

  for (auto i = 0; i < num_rois * 6; i++) {
    fout << rois[i] << " ";
  }
  fout << std::endl;
}

int main()
{
  std::string filename = "testcase";
  gen_testcase(filename);
  return 0;
}
