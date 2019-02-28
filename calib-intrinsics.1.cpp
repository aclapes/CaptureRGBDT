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
#include <boost/algorithm/string.hpp>

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
    
    std::string input_dir_list_str;
    std::string prefixes_str;
    bool verbose = false;

    po::options_description desc("Program options");
    desc.add_options()
        ("help,h", "Print help messages")
        ("corners,c", po::value<std::string>()->default_value("./corners.yml"), "")
        ("corner-selection,s", po::value<std::string>()->default_value("./corner-selection.yml"), "")
        ("intrinsics,i", po::value<std::string>()->default_value("./intrinsics.yml"), "")
        ("modality,m", po::value<std::string>()->default_value("Thermal"), "Visual modality")
        // ("preffix,F", po::value<std::string>()->default_value(""), "Image file extension")
        ("file-ext,x", po::value<std::string>()->default_value(".jpg,.jpg"), "Image file extension")
        ("pattern,p", po::value<std::string>()->default_value("8,9"), "Pattern size \"x,y\" squares")
        ("verbose,v", po::bool_switch(&verbose), "Verbosity")
        ("input-dir-list", po::value<std::string>(&input_dir_list_str)->required(), "File containing list of calibration sequence directories")
        ("prefixes", po::value<std::string>(&prefixes_str)->required(), "Prefixes");
    
    po::positional_options_description positional_options; 
    positional_options.add("input-dir-list", 1); 
    positional_options.add("prefixes", 2); 

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

    std::vector<std::string> pattern_dims;
    boost::split(pattern_dims, vm["pattern"].as<std::string>(), boost::is_any_of(","));
    assert(pattern_dims.size() == 2);

    int x = std::stoi(pattern_dims[0]);
    int y = std::stoi(pattern_dims[1]);
    assert(x > 2 && x > 2);
    cv::Size pattern_size (x,y);
    
    // ---
    
    cv::FileStorage fs;
    bool need_to_recompute = false;

    std::vector<fs::path> input_dir_list;

    std::ifstream dir_list_reader;
    dir_list_reader.open(input_dir_list_str);
    if (!dir_list_reader.is_open())
    {
        return EXIT_FAILURE;
    }
    else
    {
        std::string line;
        while (std::getline(dir_list_reader, line))
            input_dir_list.push_back(fs::path(line));
        dir_list_reader.close();
    }
    
    std::vector<std::string> prefixes;
    boost::split(prefixes, prefixes_str, boost::is_any_of(","));

    // fs::path input_dir (vm["input-dir"].as<std::string>());
    std::vector<std::vector<std::pair<fs::path,fs::path> > > sequences;
    for (fs::path input_dir : input_dir_list)
    {
        std::vector<uls::Timestamp> log1 = uls::read_log_file(input_dir / fs::path("rs.log"));
        std::vector<uls::Timestamp> log2 = uls::read_log_file(input_dir / fs::path("pt.log"));
        std::vector<std::pair<uls::Timestamp,uls::Timestamp> > log_sync;
        uls::time_sync(log1, log2, log_sync);

        std::vector<std::pair<fs::path,fs::path> > s;
        for (int i = 0; i < log_sync.size(); i++)
        {
            prefixes[0] + log_sync[i].first.id + 
        }


    }

        // std::vector<std::vector<fs::path> > s;
        // for (int i = 0; i < prefixes.size(); i++)
        // {
        //     std::vector<fs::path> frames_dir = uls::list_files_in_directory(input_dir / fs::path(prefixes[i]), vm["file-ext"].as<std::string>());
        //     std::sort(frames_dir.begin(), frames_dir.end());  // sort files by filename
        //     s.push_back(frames_dir);
        // }
        // sequences.push_back(s);
    }

    // /* Corners */

    // cv::Mat corners_all; // (frames_corners.size(), pattern_size.height * pattern_size.width * 2, CV_32F);
    // // cv::Mat fids_all; // (frames_inds.size(), 1, CV_32S);
    // std::vector<std::string> frames_all;

    // fs.open(vm["corners"].as<std::string>(), cv::FileStorage::READ);
    // if (fs.isOpened())
    // {
    //     fs["corners_mat"] >> corners_all;
    //     // fs["indices_mat"] >> fids_all;
    //     fs["frames_vec"] >> frames_all;
    // }
    // else
    // {   
    //     std::vector<cv::Mat> corners;
    //     std::vector<int> fids;
    //     for (int i = 0; i < sequences.size(); i++)
    //     {
    //         std::vector<cv::Mat> corners_;
    //         std::vector<int> fids_;
    //         std::string modality = vm["modality"].as<std::string>();
    //         if (vm["modality"].as<std::string>() == "Color")
    //             uls::find_chessboard_corners<uls::ColorFrame>(sequences[i], pattern_size, corners_, fids_, verbose);
    //         else if (vm["modality"].as<std::string>() == "Depth")
    //             uls::find_chessboard_corners<uls::DepthFrame>(sequences[i], pattern_size, corners_, fids_, verbose);
    //         else if (vm["modality"].as<std::string>() == "Thermal")
    //             uls::find_chessboard_corners<uls::ThermalFrame>(sequences[i], pattern_size, corners_, fids_, true);

    //         corners.insert(corners.end(), corners_.begin(), corners_.end());

    //         std::vector<std::string> frames_;
    //         for (int j = 0; j < fids_.size(); j++)
    //             frames_.push_back(sequences[i][fids_[j]].string());
    //         frames_all.insert(frames_all.end(), frames_.begin(), frames_.end());
    //     }

    //     corners_all.create(corners.size(), pattern_size.height * pattern_size.width * 2, CV_32F);
    //     // fids_all.create(corners.size(), 1, CV_32S);
    //     for (int j = 0; j < corners.size(); j++)
    //     {
    //         corners[j].reshape(1,1).copyTo(corners_all.row(j));
    //         // fids_all.at<int>(j,0) = fids[j];
    //     }

    //     fs.open(vm["corners"].as<std::string>(), cv::FileStorage::WRITE);
    //     fs << "corners_mat" << corners_all;
    //     // fs << "indices_mat" << fids_all;
    //     fs << "frames_vec" << frames_all;

    //     need_to_recompute = true;
    // }
    // fs.release();


    // /* Corner selection */

    // cv::Mat corners;
    // std::vector<std::string> files;
    // fs.open(vm["corner-selection"].as<std::string>(), cv::FileStorage::READ);
    // if (fs.isOpened() && !need_to_recompute)
    // {
    //     fs["corners"] >> corners;
    //     fs["files"] >> files;
    // }
    // else
    // {
    //     cv::Mat labels, centers;
    //     cv::kmeans(corners_all, 50, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

    //     // std::vector<fs::path> frame_selection (50);
    //     // std::vector<std::vector<cv::Point2f> > image_points;
    //     cv::Mat corners (50, corners_all.cols, corners_all.type());
    //     cv::Mat fids (50, 1, CV_32SC1);
    //     fids.setTo(-1);

    //     for (int i = 0; i < labels.rows; i++)
    //     {
    //         int lbl = labels.at<int>(i,0);
    //         if (fids.at<int>(lbl,0) < 0) //(frame_selection[lbl].empty())
    //         {
    //             fs::path frame_path = frames_all[i]; //frames[fids_all.at<int>(i,0)];
    //             cv::Mat img;
    //             if (vm["modality"].as<std::string>() == "Color")
    //                 img = uls::ColorFrame(frame_path).mat();
    //             else if (vm["modality"].as<std::string>() == "Thermal") 
    //                 img = uls::ThermalFrame(frame_path).mat();
    //             cv::Mat corners_img = corners_all.row(i).reshape(2, pattern_size.width * pattern_size.height);
    //             cv::drawChessboardCorners(img, pattern_size, corners_img, true);
    //             cv::imshow("Viewer", img);
    //             char ret = cv::waitKey();
    //             if (ret == ' ')
    //                 continue;
    //             else if (ret == 13) 
    //             {
    //                 // frame_selection[lbl] = frame_path;
    //                 // std::vector<cv::Point2f> corners_vec;
    //                 // for (int j = 0; j < corners.rows; j++)
    //                 //     corners_vec.push_back(cv::Point2f(corners.at<float>(j,0), corners.at<float>(j,1)));
    //                 // image_points.push_back(corners_vec);
    //                 corners_all.row(i).copyTo(corners.row(lbl));
    //                 fids.at<int>(lbl,0) = i;
    //                 files.push_back(frame_path.string());
    //             }
    //             else if (ret == 27)
    //                 return FORCED_EXIT;
    //         }
    //     }

    //     corners = uls::mask_rows(corners, fids >= 0);

    //     assert(corners.rows == files.size());

    //     fs.open(vm["corner-selection"].as<std::string>(), cv::FileStorage::WRITE);
    //     fs << "corners" << corners;
    //     fs << "files" << files;

    //     need_to_recompute = true;
    // }
    // fs.release();


    // /* Intrinsics */

    // cv::Mat camera_matrix, dist_coeffs;

    // fs.open(vm["intrinsics"].as<std::string>(), cv::FileStorage::READ);
    // if (fs.isOpened() && !need_to_recompute)
    // {
    //     fs["camera_matrix"] >> camera_matrix;
    //     fs["dist_coeffs"] >> dist_coeffs;
    // }
    // else
    // {
    //     camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    //     dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
    //     std::vector<cv::Mat> rvecs, tvecs;

    //     std::vector<std::vector<cv::Point2f> > image_points;
    //     uls::mat_to_vecvec<cv::Point2f>(corners.reshape(2, corners.rows), image_points);
        
    //     std::vector<std::vector<cv::Point3f> > object_points (1);
    //     uls::calcBoardCornerPositions(pattern_size, 0.07f, 0.05f, object_points[0]);
    //     object_points.resize(image_points.size(), object_points[0]);

    //     double rms = cv::calibrateCamera(object_points, image_points, cv::Size(640,480), camera_matrix, dist_coeffs, rvecs, tvecs);
        
    //     fs.open(vm["intrinsics"].as<std::string>(), cv::FileStorage::WRITE);
    //     fs << "camera_matrix" << camera_matrix;
    //     fs << "dist_coeffs" << dist_coeffs;

    //     need_to_recompute = true;
    // }
    // fs.release();

    // for (fs::path p : frames_all)
    // {
    //     cv::Mat img;
    //     if (vm["modality"].as<std::string>() == "Color")
    //         img = uls::ColorFrame(p).mat();
    //     else if (vm["modality"].as<std::string>() == "Thermal")
    //         img = uls::ThermalFrame(p).mat();
    //     cv::Mat tmp = img.clone();
    //     cv::undistort(tmp, img, camera_matrix, dist_coeffs);
    //     cv::imshow("Viewer", img);
    //     cv::waitKey(33);
    // }

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

