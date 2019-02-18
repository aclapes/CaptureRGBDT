// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <ctime>   // localtime
#include <sstream> // stringstream
#include <iomanip> // put_time
#include <string>  // string
#include <boost/format.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>

#include "utils.hpp"

bool debug = true;

namespace po = boost::program_options;
namespace fs = boost::filesystem;

namespace 
{ 
  const size_t ERROR_IN_COMMAND_LINE = 1; 
  const size_t SUCCESS = 0; 
  const size_t ERROR_UNHANDLED_EXCEPTION = 2; 
 
} // namespace

cv::Mat get_reference_corners(cv::Size pattern_size)
{
    cv::Mat corners_ref (pattern_size.height*pattern_size.width, 2, CV_32FC1);

    float w_step = 1.f / (pattern_size.height-1);
    float h_step = 1.f / (pattern_size.width-1);

    for (int j = 0; j < pattern_size.height; j++)
    {
        for (int i = 0; i < pattern_size.width; i++)
        {
            corners_ref.at<float>(j*pattern_size.width+i, 0) = j*w_step;
            corners_ref.at<float>(j*pattern_size.width+i, 1) = 1.f - i*h_step;
        }
    }

    return corners_ref;
}

bool check_transformed_corners(cv::Mat corners, cv::Size pattern_size, cv::Size2f eps = cv::Size2f(0.f,0.f))
{    
    float w_step = 1.f / (pattern_size.height-1);
    float h_step = 1.f / (pattern_size.width-1);

    if (eps.height == 0.f)
        eps.height = w_step / 4.f;
    if (eps.width == 0.f)
        eps.width = h_step / 4.f;

    for (int j = 0; j < pattern_size.height; j++)
    {
        for (int i = 0; i < pattern_size.width; i++)
        {
            float diff_j = abs(corners.at<float>(j*pattern_size.width+i, 0) - j*w_step);
            float diff_i = abs(corners.at<float>(j*pattern_size.width+i, 1) - (1.f - i*h_step));
            if (diff_j > eps.height || diff_i > eps.width)
                return false;
        }
    }

    return true;
}

int main(int argc, char * argv[]) try
{
    /* -------------------------------- */
    /*   Command line argument parsing  */
    /* -------------------------------- */
    
    std::string input_dir_str;

    po::options_description desc("Program options");
    desc.add_options()
        ("help,h", "Print help messages")
        ("file-ext,x", po::value<std::string>()->default_value(".jpg"), "Image file extension")
        ("input-dir", po::value<std::string>(&input_dir_str)->required(), "Input directory containing pt frames and timestamp files");
    
    po::positional_options_description positional_options; 
    positional_options.add("input-dir", 1); 

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm); // can throw

    // --help option?
    if (vm.count("help"))
    {
        std::cout << "Basic Command Line Parameter App" << std::endl
        << desc << std::endl;
        return SUCCESS;
    }
    
    po::notify(vm); // throws on error, so do after help in case
    
    /* --------------- */
    /*    Main code    */
    /* --------------- */
        
    fs::path input_dir (vm["input-dir"].as<std::string>());

    std::vector<fs::path> frames = uls::list_files_in_directory(input_dir, vm["file-ext"].as<std::string>());
    std::sort(frames.begin(), frames.end());  // sort files by filename

    cv::Size pattern_size (9,8); // (width, height)

    float tracking_enabled = false;
    int nb_tracked_frames;
    cv::Mat corners, corners_prev;
    cv::Mat img, img_prev;
    for (int i = 0; i < frames.size(); i++) 
    {
        std::string img_file_path = frames[i].string();
        img = cv::imread(img_file_path, CV_LOAD_IMAGE_UNCHANGED);
        img = uls::thermal_to_8bit(img);
        // cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        std::cout << img_file_path << '\n';
        cv::resize(img,img,cv::Size(640,480));
        // cv::imshow("Viewer", img);
        // cv::waitKey(0);S

        bool patternfound_9x8 = findChessboardCorners(img, pattern_size, corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
        if (patternfound_9x8) {
            tracking_enabled = true;
            // nb_tracked_frames = 0;
        }
        else if (tracking_enabled)
        {
            cv::Mat status, err;
            cv::calcOpticalFlowPyrLK(img_prev, img, corners_prev, corners, status, err, cv::Size(7,7));

            if (pattern_size.width * pattern_size.height != cv::sum(status)[0])
                tracking_enabled = false;
        }

        // if (nb_tracked_frames > 15) 
        //     tracking_enabled = false;

        cv::Mat cimg;
        cv::cvtColor(img, cimg, cv::COLOR_GRAY2BGR);

        if (tracking_enabled)
        {
            cornerSubPix(img, corners, cv::Size(15, 15), cv::Size(5, 5), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

            cv::Mat corners_ref = get_reference_corners(pattern_size);

            cv::Mat mask;
            cv::Mat h = cv::findHomography(corners, corners_ref, mask, CV_RANSAC);

            cv::Mat corners_transf;
            cv::perspectiveTransform(corners, corners_transf, h);

            if ( check_transformed_corners(corners_transf, pattern_size) )
                cv::drawChessboardCorners(cimg, cv::Size(9,8), corners, patternfound_9x8);
            else
                tracking_enabled = false;
        }
 
        cv::imshow("Viewer", cimg);
        cv::waitKey(33);

        img_prev = img;
        corners_prev = corners;
    }

    return SUCCESS;
}
catch(po::error& e)
{
    std::cerr << "Error parsing arguments: " << e.what() << std::endl;
    return ERROR_IN_COMMAND_LINE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return ERROR_UNHANDLED_EXCEPTION;
}

