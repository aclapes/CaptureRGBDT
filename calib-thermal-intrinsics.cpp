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
#include <boost/progress.hpp>

#include "utils.hpp"

bool debug = true;

namespace po = boost::program_options;
namespace fs = boost::filesystem;

namespace 
{ 
  const size_t ERROR_IN_COMMAND_LINE = 1; 
  const size_t SUCCESS = 0; 
  const size_t ERROR_UNHANDLED_EXCEPTION = 2; 
  const size_t FORCED_EXIT = 3;
 
} // namespace

int main(int argc, char * argv[]) try
{
    /* -------------------------------- */
    /*   Command line argument parsing  */
    /* -------------------------------- */
    
    std::string input_dir_str;
    bool verbose = false;

    po::options_description desc("Program options");
    desc.add_options()
        ("help,h", "Print help messages")
        ("corners,c", po::value<std::string>()->default_value("./corners.yml"), "")
        ("corner-selection,s", po::value<std::string>()->default_value("./corner-selection.yml"), "")
        ("intrinsics,i", po::value<std::string>()->default_value("./intrinsics.yml"), "")
        ("modality,m", po::value<std::string>()->default_value("thermal"), "Visual modality")
        ("file-ext,x", po::value<std::string>()->default_value(".jpg"), "Image file extension")
        ("verbose,v", po::bool_switch(&verbose), "Verbosity")
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

    cv::Size pattern_size (8,9); // (width, height)
    cv::FileStorage fs;
    bool need_to_recompute = false;

    fs::path input_dir (vm["input-dir"].as<std::string>());
    std::vector<fs::path> frames = uls::list_files_in_directory(input_dir, vm["file-ext"].as<std::string>());
    std::sort(frames.begin(), frames.end());  // sort files by filename


    /* Corners */

    cv::Mat corners_all; // (frames_corners.size(), pattern_size.height * pattern_size.width * 2, CV_32F);
    cv::Mat fids_all; // (frames_inds.size(), 1, CV_32S);

    fs.open(vm["corners"].as<std::string>(), cv::FileStorage::READ);
    if (fs.isOpened())
    {
        fs["corners_mat"] >> corners_all;
        fs["indices_mat"] >> fids_all;
    }
    else
    {
        std::vector<cv::Mat> corners_;
        std::vector<int> fids_;
        std::string modality = vm["modality"].as<std::string>();
        if (modality == "color")
            uls::find_chessboard_corners<uls::ColorFrame>(frames, pattern_size, corners_, fids_, verbose);
        else if (modality == "depth")
            uls::find_chessboard_corners<uls::DepthFrame>(frames, pattern_size, corners_, fids_, verbose);
        if (modality == "thermal")
            uls::find_chessboard_corners<uls::ThermalFrame>(frames, pattern_size, corners_, fids_, verbose);
        corners_all.create(corners_.size(), pattern_size.height * pattern_size.width * 2, CV_32F);
        fids_all.create(fids_.size(), 1, CV_32S);
        for (int i = 0; i < corners_.size(); i++)
        {
            corners_[i].reshape(1,1).copyTo(corners_all.row(i));
            fids_all.at<int>(i,0) = fids_[i];
        }

        fs.open(vm["corners"].as<std::string>(), cv::FileStorage::WRITE);
        fs << "corners_mat" << corners_all;
        fs << "indices_mat" << fids_all;

        need_to_recompute = true;
    }
    fs.release();


    /* Corner selection */

    cv::Mat corners, fids;
    fs.open(vm["corner-selection"].as<std::string>(), cv::FileStorage::READ);
    if (fs.isOpened() && !need_to_recompute)
    {
        fs["corners"] >> corners;
    }
    else
    {
        cv::Mat labels, centers;
        cv::kmeans(corners_all, 50, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

        // std::vector<fs::path> frame_selection (50);
        // std::vector<std::vector<cv::Point2f> > image_points;
        cv::Mat corners (50, corners_all.cols, corners_all.type());
        cv::Mat fids (50, 1, CV_32SC1);
        fids.setTo(-1);

        for (int i = 0; i < labels.rows; i++)
        {
            int lbl = labels.at<int>(i,0);
            if (fids.at<int>(lbl,0) < 0) //(frame_selection[lbl].empty())
            {
                fs::path frame_path = frames[fids_all.at<int>(i,0)];
                std::cout << frame_path << '\n';
                cv::Mat img = uls::ThermalFrame(frame_path).mat();
                cv::Mat corners_img = corners_all.row(i).reshape(2, pattern_size.width * pattern_size.height);
                cv::drawChessboardCorners(img, pattern_size, corners_img, true);
                cv::imshow("Viewer", img);
                char ret = cv::waitKey();
                if (ret == ' ')
                    continue;
                else if (ret == 13) 
                {
                    // frame_selection[lbl] = frame_path;
                    // std::vector<cv::Point2f> corners_vec;
                    // for (int j = 0; j < corners.rows; j++)
                    //     corners_vec.push_back(cv::Point2f(corners.at<float>(j,0), corners.at<float>(j,1)));
                    // image_points.push_back(corners_vec);
                    corners_all.row(i).copyTo(corners.row(lbl));
                    fids.at<int>(lbl,0) = i;
                }
                else if (ret == 27)
                    return FORCED_EXIT;
            }
        }

        corners = uls::mask_rows(corners, fids >= 0);

        fs.open(vm["corner-selection"].as<std::string>(), cv::FileStorage::WRITE);
        fs << "corners" << corners;

        need_to_recompute = true;
    }
    fs.release();


    /* Intrinsics */

    cv::Mat camera_matrix, dist_coeffs;

    fs.open(vm["intrinsics"].as<std::string>(), cv::FileStorage::READ);
    if (fs.isOpened() && !need_to_recompute)
    {
        fs["camera_matrix"] >> camera_matrix;
        fs["dist_coeffs"] >> dist_coeffs;
    }
    else
    {
        camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
        std::vector<cv::Mat> rvecs, tvecs;

        std::vector<std::vector<cv::Point2f> > image_points;
        uls::mat_to_vecvec<cv::Point2f>(corners.reshape(2, corners.rows), image_points);
        
        std::vector<std::vector<cv::Point3f> > object_points (1);
        uls::calcBoardCornerPositions(pattern_size, 0.07f, 0.05f, object_points[0]);
        object_points.resize(image_points.size(), object_points[0]);

        double rms = cv::calibrateCamera(object_points, image_points, cv::Size(640,480), camera_matrix, dist_coeffs, rvecs, tvecs);
        
        fs.open(vm["intrinsics"].as<std::string>(), cv::FileStorage::WRITE);
        fs << "camera_matrix" << camera_matrix;
        fs << "dist_coeffs" << dist_coeffs;

        need_to_recompute = true;
    }
    fs.release();

    for (fs::path p : frames)
    {
        cv::Mat img = uls::ThermalFrame(p).mat();
        cv::Mat tmp = img.clone();
        cv::undistort(tmp, img, camera_matrix, dist_coeffs);
        cv::imshow("Viewer", img);
        cv::waitKey(33);
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

