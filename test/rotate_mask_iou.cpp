#include <iostream>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <cmath>

#include <omp.h>

#define PRINT(a) std::cout << #a << ": " << a << std::endl;

template <typename T>
inline T deg2rad(const T deg)
{
    return deg / 180.0 * 3.1415926535;
}

void get_rotated_roi_pixel_mapping(float* M, const float* roi)
{

	float xc = roi[0];
	float yc = roi[1];
	float w = roi[2];
	float h = roi[3];
	float angle = roi[4];

    float center[2] {xc, yc};
    float theta = deg2rad(angle);

    // paste mask onto image via rotated rect mapping
    float v_x[2] {cos(theta), sin(theta)};
    float v_y[2] {-sin(theta), cos(theta)};
    float s_x = center[0] - v_x[0] * ((w - 1) / 2) - v_y[0] * ((h - 1) / 2);
    float s_y = center[1] - v_x[1] * ((w - 1) / 2) - v_y[1] * ((h - 1) / 2);

	M[0] = v_x[0];
	M[1] = v_y[0];
	M[2] = s_x;
	M[3] = v_x[1];
	M[4] = v_y[1];
	M[5] = s_y;
}

void paste_rotated_roi_in_image(cv::Mat& out, const cv::Mat& image, const cv::Mat& roi_image, const float* roi)
{
	int im_w = image.cols;
	int im_h = image.rows;
	if (im_w == 0 || im_h == 0)
		return;
	
	out = cv::Mat(im_h, im_w, image.type());

	int h = round(roi[3]); 
	int w = round(roi[2]); 

    int rh = roi_image.rows;
	int rw = roi_image.cols;

    if (rw != w || rh != h)
	{
		cv::resize(roi_image, roi_image, cv::Size(w, h));
	}

    // generate the mapping of points from roi_image to an image
	float M[6];
    get_rotated_roi_pixel_mapping(M, roi);

	for(size_t y = 0; y < h; y++)
	{
		for(size_t x = 0; x < w; x++)
		{
			int mapped_x = x * M[0] + y * M[1] + M[2];
			int mapped_y = x * M[3] + y * M[4] + M[5];
			if (mapped_x >= 0 && mapped_x < im_w && mapped_y >= 0 && mapped_y < im_h)
			{
				out.at<uchar>(mapped_y, mapped_x) = roi_image.at<uchar>(y, x); // TODO: remove uchar
			}
		}
	}

	// perform hole fill on mask
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
	cv::morphologyEx(out, out, cv::MORPH_CLOSE, kernel);
}

float compute_rotate_mask_iou(const cv::Mat& gt_mask, const float* proposal)
{
	// proposal: xc yc w h angle
	int img_h = gt_mask.rows;
	int img_w = gt_mask.cols;

	int h = round(proposal[3]); 
	int w = round(proposal[2]); 

	cv::Mat proposal_mask(h, w, CV_8UC1);
	proposal_mask += 1;

	cv::Mat paste_mask;
	paste_rotated_roi_in_image(paste_mask, gt_mask, proposal_mask, proposal);

	// cv::imshow("paste_mask", paste_mask*255);
	// cv::waitKey(0);

	cv::Mat out_mask;
	// cv::bitwise_and(gt_mask, paste_mask, out_mask);
	out_mask = gt_mask & paste_mask;

	float full_area = cv::sum(gt_mask)[0];

	if (full_area == 0)
		return 0.0f;

	float box_area = cv::sum(out_mask)[0];
	float mask_iou = box_area / full_area;
	return mask_iou;
}


int main(int argc, char const *argv[])
{
	int ITERS = 1000;
	int img_h = 800;
    int img_w = 800;

	float proposal[5] = {img_h/2, img_w/2, img_h/3, img_w/3, 30};
	
	cv::Mat gt_mask(img_h, img_w, CV_8UC1); 
	for(size_t y = img_h / 4; y < img_h/4*3; y++)
	{
		for(size_t x = img_w/4; x < img_w/4*3; x++)
		{
			gt_mask.at<uchar>(y, x) = 1;
		}
	}

	std::clock_t start;

    start = std::clock();

	// #pragma omp parallel for
	for(size_t i = 0; i < ITERS; i++)
	{
		float mask_iou = compute_rotate_mask_iou(gt_mask, proposal);
		// printf("mask_iou: %.3f\n", mask_iou);
	}

    double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    printf("duration: %.3f\n", duration);

	return 0;
}