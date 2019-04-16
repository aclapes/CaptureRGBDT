//
//  calibration.hpp
//
//  Created by Albert Clapés on 16/04/2018.
//

#ifndef utils_calibration_hpp
#define utils_calibration_hpp

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>

namespace fs = boost::filesystem;

namespace uls
{
    typedef struct
    {
        cv::Mat camera_matrix;
        cv::Mat dist_coeffs;
    } intrinsics_t;

    typedef struct
    {
        cv::Mat R;
        cv::Mat T;
    } extrinsics_t;

    void camera_matrix_values(cv::Mat K, double & fx, double & fy, double & cx, double & cy)
    {
        fx = K.at<double>(0,0);
        fy = K.at<double>(1,1);
        cx = K.at<double>(0,2);
        cy = K.at<double>(1,2);
    }

    void align_to_depth(cv::Mat depth, 
                        cv::Mat K_depth,
                        cv::Mat K_other,
                        float depth_scale,
                        std::shared_ptr<extrinsics_t> extrinsics,
                        cv::Mat & map_x, 
                        cv::Mat & map_y)
    {
        double fx_d, fy_d, cx_d, cy_d, fx_o, fy_o, cx_o, cy_o;
        uls::camera_matrix_values(K_depth, fx_d, fy_d, cx_d, cy_d);
        uls::camera_matrix_values(K_other, fx_o, fy_o, cx_o, cy_o);

        cv::Mat R = extrinsics->R;
        cv::Mat T = extrinsics->T;

        double x, y, z;
        double p_x, p_y, p_z;

        map_x.release();
        map_y.release();

        map_x.create(depth.size(), CV_32FC1);
        map_y.create(depth.size(), CV_32FC1);

        for (int i = 0; i < depth.rows; i++)
        {
            for (int j = 0; j < depth.cols; j++)
            {
                z = depth_scale * depth.at<unsigned short>(i,j);
                if (z > 0)
                {
                    x = (j - cx_d) * z / fx_d;
                    y = (i - cy_d) * z / fy_d;

                    p_x = (R.at<double>(0,0) * x + R.at<double>(0,1) * y + R.at<double>(0,2) * z) + T.at<double>(0,0);
                    p_y = (R.at<double>(1,0) * x + R.at<double>(1,1) * y + R.at<double>(1,2) * z) + T.at<double>(1,0);
                    p_z = (R.at<double>(2,0) * x + R.at<double>(2,1) * y + R.at<double>(2,2) * z) + T.at<double>(2,0);

                    map_x.at<float>(i,j) = (p_x * fx_o / p_z) + cx_o;
                    map_y.at<float>(i,j) = (p_y * fy_o / p_z) + cy_o;
                }
            }
        }
    }
}

#endif /* utils_calibration_hpp */