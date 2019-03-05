#include "pch.h"
#include <iostream>
#include "opencv2/opencv.hpp"
#include <vector>
#include <utility>


using namespace cv;

using LabelType = std::vector<std::vector<int>>;
using SegmentType = std::vector<std::vector<std::vector<int>>>;

LabelType img_full(int heigth, int width, int filler=-1) {
	return LabelType(heigth, std::vector<int>(width, filler));
}

SegmentType img_zero(int heigth, int width, int ch_count) {
	return SegmentType(heigth, LabelType(width, std::vector<int>(ch_count * 2 + 1, 0)));
}
std::vector<int> minimum(std::vector<int> left_arr, std::vector<int> rigth_arr) {
	std::vector<int> result;
	result.reserve(left_arr.size());
	for (auto i = 0; i < left_arr.size(); i++) {
		result.push_back(std::min(left_arr[i], rigth_arr[i]));
	}
	return result;
}

std::vector<int> maximum(std::vector<int> left_arr, std::vector<int> rigth_arr) {
	std::vector<int> result;
	result.reserve(left_arr.size());
	for (auto i = 0; i < left_arr.size(); i++) {
		result.push_back(std::max(left_arr[i], rigth_arr[i]));
	}
	return result;
}

std::vector<int> subtract(std::vector<int> left, std::vector<int> right) {
	std::vector<int> difference;
	std::set_difference(
		left.begin(), left.end(),
		right.begin(), right.end(),
		std::back_inserter(difference)
	);
	return difference;
}

std::vector<int> hstack(int num, std::vector<int> vec1, std::vector<int> vec2) {
	std::vector<int> result;
	result.reserve(vec1.size() + vec2.size() + 1);
	result.push_back(num);
	result.insert(vec1.begin(), vec1.end(), result.end());
	result.insert(vec2.begin(), vec2.end(), result.end());
	return result;
}

//SegmentType vstack(SegmentType segment, SegmentType vec_seg) {
//
//}

std::pair<LabelType, SegmentType> extract_superpixel(Mat img, int eps) {
	int ch_count = 1;
	auto label = img_full(img.cols , img.rows, -1);
	auto segments = img_zero(1, 1, ch_count);
	auto vector_segment = img_zero(1, 1, ch_count);
	auto segment_num = 0;

	for (int i = 0; i < img.cols; i++) {
		for (int j = 0; j < img.rows; j++) {			
			// Vec3b intensity = img.at<Vec3b>(y, x);
			// uchar blue = intensity.val[0];
			// uchar green = intensity.val[1];
			// uchar red = intensity.val[2];
			
			auto im_mat_vector = img.at<Vec3b>(i, j);
			std::vector<int> im_vector = { im_mat_vector[0], im_mat_vector[1], im_mat_vector[2] };
			if (i > 0 && j > 0) {
				auto left_label = label[i][j-1];
				auto upper_label = label[i - 1][j];
				auto new_label_left = segments[left_label][0][0];
				auto new_label_up = segments[upper_label][0][0];
				auto needed_segment_left = segments[new_label_left][0];
				auto needed_segment_upper = segments[new_label_up][0];
				
				auto min_seg_new_left = minimum(
					std::vector<int>(needed_segment_left.begin() + 1,
						needed_segment_left.begin() + ch_count + 1), im_vector);
				auto max_seg_new_left = maximum(
					std::vector<int>(needed_segment_left.begin() + ch_count,
						needed_segment_left.end()), im_vector);

				auto min_seg_new_up = minimum(
					std::vector<int>(needed_segment_upper.begin() + 1,
						needed_segment_upper.begin() + ch_count + 1), im_vector);
				auto max_seg_new_up = maximum(
					std::vector<int>(needed_segment_upper.begin() + ch_count,
						needed_segment_upper.end()), im_vector);

				auto conditional_vec_up = subtract(max_seg_new_up, min_seg_new_up);
				auto conditional_vec_left = subtract(max_seg_new_left, min_seg_new_left);

				auto condition_up = std::all_of(conditional_vec_up.begin(),
					conditional_vec_up.end(),
					[eps](int elem) {return elem <= 2 * eps; });
				auto condition_left = std::all_of(conditional_vec_left.begin(),
					conditional_vec_left.end(),
					[eps](int elem) {return elem <= 2 * eps; });


				if (new_label_left == new_label_up) {
					if (condition_up){
						label[i][j] = new_label_up;
						for (int h = ch_count + 1; h < max_seg_new_up.size(); h++) {
							segments[new_label_up][0][h] = max_seg_new_up[h - ch_count - 1];
						}
						for (int h = 1; h < ch_count+1; h++) {
							segments[new_label_up][0][h] = min_seg_new_up[h - 1];
						}						
					}
					else {
						segment_num++;
						label[i][j] = segment_num;
						vector_segment[0][0] = hstack(segment_num, im_vector, im_vector);
					}
				}
			}
		}
	}
	return std::pair(label, segments);
}


int main(int, char**)
{
	VideoCapture cap("C:\\NIRS\\video.mp4"); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	for (;;)
	{
		Mat frame;
		cap >> frame; // get a new frame from camera
		auto ret = extract_superpixel(frame, 10);
		imshow("frame", frame);
		if (waitKey(30) >= 0) break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}