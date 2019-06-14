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

#include "utils/common.hpp"
#include "utils/calibration.hpp"

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

// static double computeReprojectionErrors( const vector<vector<Point3f> >& objectPoints,
//                                          const vector<vector<Point2f> >& imagePoints,
//                                          const vector<Mat>& rvecs, const vector<Mat>& tvecs,
//                                          const Mat& cameraMatrix , const Mat& distCoeffs,
//                                          vector<float>& perViewErrors)
// {
//     vector<Point2f> imagePoints2;
//     int i, totalPoints = 0;
//     double totalErr = 0, err;
//     perViewErrors.resize(objectPoints.size());

//     for( i = 0; i < (int)objectPoints.size(); ++i )
//     {
//         projectPoints( Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
//                        distCoeffs, imagePoints2);
//         err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);

//         int n = (int)objectPoints[i].size();
//         perViewErrors[i] = (float) std::sqrt(err*err/n);
//         totalErr        += err*err;
//         totalPoints     += n;
//     }

//     return std::sqrt(totalErr/totalPoints);
// }

int main(int argc, char * argv[]) try
{
    /* -------------------------------- */
    /*   Command line argument parsing  */
    /* -------------------------------- */
    
    std::string input_dir_list_str;
    // std::string prefixes_str;
    bool verbose = false;
    std::string corners_file;

    po::options_description desc("Program options");
    desc.add_options()
        ("help,h", "Print help messages")
        ("input-corners-file", po::value<std::string>(&corners_file)->required(), "Input corners file")
        ("parent-dir", po::value<std::string>()->default_value(""), "Path containing the sequences")
        // ("corners,c", po::value<std::string>()->default_value("./corners.yml"), "")
        // ("corner-selection,s", po::value<std::string>()->default_value("./corner-selection.yml"), "")
        // ("intrinsics,i", po::value<std::string>()->default_value("./intrinsics.yml"), "")
        // ("modality,m", po::value<std::string>()->default_value("Thermal"), "Visual modality")
        // ("preffix,F", po::value<std::string>()->default_value(""), "Image file extension")
        // ("file-ext,x", po::value<std::string>()->default_value(".jpg,.jpg"), "Image file extension")
        // ("pattern,p", po::value<std::string>()->default_value("8,9"), "Pattern size \"x,y\" squares")
        // ("verbose,v", po::bool_switch(&verbose), "Verbosity")
        // ("input-dir-list", po::value<std::string>(&input_dir_list_str)->required(), "File containing list of calibration sequence directories")
        // ("prefixes", po::value<std::string>(&prefixes_str)->required(), "Prefixes");
        ("square-size,q", po::value<std::string>()->default_value("0.05,0.05"), "Square size in meters")
        ("intrinsics-file", po::value<std::string>()->default_value(""), "Intrinsics")
        ("nb-clusters,k", po::value<int>()->default_value(50), "Number of k-means clusters");

    po::positional_options_description positional_options; 
    positional_options.add("input-corners-file", 1);
    // positional_options.add("input-dir-list", 1); 
    // positional_options.add("prefixes", 2); 

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

    cv::FileStorage corners_fs (corners_file, cv::FileStorage::READ);

    int nb_sequence_dirs;
    corners_fs["nb_sequences"] >> nb_sequence_dirs;
    
    std::vector<std::string> frames_all;
    for (int i = 0; i < nb_sequence_dirs; i++)
    {
        std::string sequence_dir;
        corners_fs["sequence_dir-" + std::to_string(i)] >> sequence_dir;

        fs::path sequence_path (sequence_dir);
        if (sequence_path.is_relative())
            sequence_path = fs::path(vm["parent-dir"].as<std::string>()) / sequence_path;

        std::vector<std::string> frames;
        corners_fs["frames-" + std::to_string(i)] >> frames;
        for (int j = 0; j < frames.size(); j++)
            frames[j] = (fs::path(sequence_dir) / fs::path(frames[j])).string();
            
        frames_all.insert(frames_all.end(), frames.begin(), frames.end());
    }

    cv::Size pattern_size;
    corners_fs["pattern_size"] >> pattern_size;

    cv::Mat corners_all (frames_all.size(), pattern_size.height * pattern_size.width * 2, CV_32FC1);
    int corners_count = 0;
    for (int i = 0; i < nb_sequence_dirs; i++)
    {
        cv::Mat corners_aux;
        corners_fs["corners-" + std::to_string(i)] >> corners_aux;
        cv::Mat c = corners_aux.reshape(1, corners_aux.rows);
        std::cout << c.rows << "," << c.cols << "," << c.channels() << std::endl;
        if (c.rows > 0)
            c.copyTo(corners_all(cv::Rect(0, corners_count, pattern_size.height * pattern_size.width * 2, corners_aux.rows)));
        corners_count += corners_aux.rows;
    }

    int K = vm["nb-clusters"].as<int>();

    cv::Mat labels, centers;
    cv::kmeans(corners_all, K, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

    // cv::Mat corners (K, corners_all.cols, corners_all.type());
    // cv::Mat fids (K, 1, CV_32SC1);
    // fids.setTo(-1);

    std::vector<std::string> square_size_aux;
    boost::split(square_size_aux, vm["square-size"].as<std::string>(), boost::is_any_of(","));

    assert(square_size_aux.size() == 1 || square_size_aux.size() == 2);
    float square_width  = std::stof(square_size_aux[0]);
    float square_height = std::stof(square_size_aux[square_size_aux.size()-1]); // obscure hack

    assert(square_width > 0 && square_height > 0);
    cv::Point2f square_size (square_width, square_height);

    cv::Size frame_size;
    corners_fs["resize_dims"] >> frame_size;

    std::vector<std::vector<int> > indices;
    for (int k = 0; k < K; k++)
    {
        std::vector<int> indices_k;
        for (int i = 0; i < labels.rows; i++)
            if (labels.at<int>(i,0) == k) indices_k.push_back(i);

        auto rng = std::default_random_engine {};
        std::shuffle(indices_k.begin(), indices_k.end(), rng);
        indices.push_back(indices_k);
    }
    
    std::vector<cv::Mat> corners (K);
    std::vector<std::string> corner_frames (K);
    int k = 0;
    std::vector<int> ptr (K);
    bool keep_selecting = true;

    cv::namedWindow("Viewer");

    // int y_shift;
    // corners_fs["y-shift"] >> y_shift;

    std::string prefix;
    corners_fs["prefix"] >> prefix;
    boost::trim_if(prefix, &uls::is_bar);  // remove leading and ending '/' and '\' bars

    while (keep_selecting)
    {
        std::cout << k << ":" << ptr[k]+1 << "/" << indices[k].size() << std::endl;
        int idx = indices[k][ptr[k]];//indices_k.at<int>(i,0);
        // cv::Mat img;
        // if (prefix == "rs/color,rs/depth")
        //     img = uls::ColorFrame(fs::path(frames_all[idx]), frame_size).get();
        // else if (modality == "pt/thermal") 
        //     img = uls::ThermalFrame(fs::path(frames_all[idx]), frame_size).get();
        cv::Mat img = cv::imread(frames_all[idx], cv::IMREAD_UNCHANGED);
        if (prefix == "pt/thermal")
            uls::thermal_to_8bit(img, img);
        uls::resize(img, img, frame_size);

        cv::Mat corners_aux = corners_all.row(idx).reshape(2, pattern_size.width * pattern_size.height);
        // cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
        cv::drawChessboardCorners(img, pattern_size, corners_aux, true);
        std::stringstream ss;
        if (corner_frames[k] == frames_all[idx])
            ss << "[*" << k << "*]";
        else
            ss << "[ " << k << " ]";
        ss << ' ' << ptr[k] << '/' << indices[k].size(); 
        cv::putText(img, ss.str(), cv::Point(frame_size.width/20.0,frame_size.height/10.0), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,0,255), 1, 8, false);
        cv::imshow("Viewer", img);

        char ret = cv::waitKey();
        if (ret == 'j')
            ptr[k] = (ptr[k] > 0) ? ptr[k] - 1 : ptr[k];
        else if (ret == ';')
            ptr[k] = (ptr[k] < (indices[k].size() - 1)) ? ptr[k] + 1 : ptr[k];
        else if (ret == 'k')
        {
            k = (k > 0) ? k - 1 : K-1;
        }
        else if (ret == 'l' || ret == ' ')
            k = (k < (K - 1)) ? k + 1 : 0;
        else if (ret == 13) 
        {
            if (corner_frames[k] == frames_all[idx])
            {
                corner_frames[k] = std::string();
                corners[k] = cv::Mat();
            }
            else
            {
                corners[k] = corners_all.row(idx);
                corner_frames[k] = frames_all[idx];
                k = (k < (K - 1)) ? k + 1 : 0;
            }
        }
        else if (ret == 27)
            keep_selecting = false;
    }
    
    cv::destroyWindow("Viewer");

    cv::Mat corners_selection;
    std::vector<std::string> corner_frames_selection;

    for (int k = 0; k < corner_frames.size(); k++)
    {
        if (!corner_frames[k].empty())
        {
            corners_selection.push_back(corners[k]);
            corner_frames_selection.push_back(corner_frames[k]);
        }
    }
    
    /* Intrinsics */

    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
    std::vector<cv::Mat> rvecs, tvecs;

    std::vector<std::vector<cv::Point2f> > image_points;
    uls::mat_to_vecvec<cv::Point2f>(corners_selection.reshape(2, corners_selection.rows), image_points);

    std::vector<std::vector<cv::Point3f> > object_points (1);
    uls::calcBoardCornerPositions(pattern_size, square_size.x, square_size.y, object_points[0]);
    object_points.resize(image_points.size(), object_points[0]);

    double rms = cv::calibrateCamera(object_points, image_points, frame_size, camera_matrix, dist_coeffs, rvecs, tvecs);
    std::cout << "RMS: " << rms << '\n';


    if (!vm["intrinsics-file"].as<std::string>().empty())
    {
        cv::FileStorage fstorage_out (vm["intrinsics-file"].as<std::string>(), cv::FileStorage::WRITE);
        fstorage_out << "corners" << corners_selection;
        fstorage_out << "corner_frames" << corner_frames_selection;
        fstorage_out << "camera_matrix" << camera_matrix;
        fstorage_out << "dist_coeffs" << dist_coeffs;
        fstorage_out << "rms" << rms;
        fstorage_out << "resize_dims" << frame_size;
        fstorage_out << "pattern_size" << pattern_size;
        fstorage_out << "square_size" << square_size;
        // fstorage_out << "y-shift" << y_shift;
        fstorage_out.release();
    }

    corners_fs.release();

    for (std::string frame_path : frames_all)
    {
        // cv::Mat img;
        // if (prefix == "Color")
        //     img = uls::ColorFrame(fs::path(frame_path), frame_size).get();
        // else if (prefix == "Thermal")
        //     img = uls::ThermalFrame(fs::path(frame_path), frame_size).get();
        cv::Mat img = cv::imread(frame_path, cv::IMREAD_UNCHANGED);
        uls::resize(img, img, frame_size);

        cv::Mat tmp = img.clone();
        cv::undistort(tmp, img, camera_matrix, dist_coeffs);
        cv::imshow("Viewer", img);

        char ret = cv::waitKey();
        if (ret == 27)
            break;
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

