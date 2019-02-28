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
    std::string prefixes_str;
    bool verbose = false;
    std::string corners_file;

    po::options_description desc("Program options");
    desc.add_options()
        ("help,h", "Print help messages")
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
        ("intrinsics-file", po::value<std::string>()->default_value(""), "Intrinsics")
        ("nb-clusters,k", po::value<int>()->default_value(50), "Number of k-means clusters")
        ("input-corners-file", po::value<std::string>(&corners_file)->required(), "Input corners file");
    
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
    corners_fs["sequence_dirs"] >> nb_sequence_dirs;
    
    std::vector<std::string> frames_all;
    for (int i = 0; i < nb_sequence_dirs; i++)
    {
        std::string sequence_dir;
        corners_fs["sequence_dir-" + std::to_string(i)] >> sequence_dir;

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

    while (keep_selecting)
    {
        std::cout << k << ":" << ptr[k]+1 << "/" << indices[k].size() << std::endl;
        int idx = indices[k][ptr[k]];//indices_k.at<int>(i,0);
        cv::Mat img;
        if (corners_fs["modality"] == "Color")
            img = uls::ColorFrame(fs::path(frames_all[idx]), frame_size).mat();
        else if (corners_fs["modality"] == "Thermal") 
            img = uls::ThermalFrame(fs::path(frames_all[idx]), frame_size).mat();

        cv::Mat corners_aux = corners_all.row(idx).reshape(2, pattern_size.width * pattern_size.height);
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
        cv::drawChessboardCorners(img, pattern_size, corners_aux, true);
        std::stringstream ss;
        if (corner_frames[k] == frames_all[idx])
            ss << "[*" << k << "*]";
        else
            ss << "[ " << k << " ]";
        ss << ' ' << ptr[k] << '/' << indices[k].size(); 
        cv::putText(img, ss.str(), cv::Point(frame_size.width/20.0,frame_size.height/10.0), CV_FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,0,255), 1, 8, false);
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
    uls::calcBoardCornerPositions(pattern_size, 0.05f, 0.05f, object_points[0]);
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
        fstorage_out.release();
    }

    for (std::string frame_path : frames_all)
    {
        cv::Mat img;
        if (corners_fs["modality"] == "Color")
            img = uls::ColorFrame(fs::path(frame_path), frame_size).mat();
        else if (corners_fs["modality"] == "Thermal")
            img = uls::ThermalFrame(fs::path(frame_path), frame_size).mat();

        cv::Mat tmp = img.clone();
        cv::undistort(tmp, img, camera_matrix, dist_coeffs);
        cv::imshow("Viewer", img);

        char ret = cv::waitKey();
        if (ret == 27)
            break;
    }

    // corners_fs.release();

    
    corners_fs.release();

    // std::vector<std::string> pattern_dims;
    // boost::split(pattern_dims, vm["pattern"].as<std::string>(), boost::is_any_of(","));
    // assert(pattern_dims.size() == 2);

    // int x = std::stoi(pattern_dims[0]);
    // int y = std::stoi(pattern_dims[1]);
    // assert(x > 2 && x > 2);
    // cv::Size pattern_size (x,y);
    
    // // ---
    
    // cv::FileStorage fs;
    // bool need_to_recompute = false;

    // std::vector<fs::path> input_dir_list;

    // std::ifstream dir_list_reader;
    // dir_list_reader.open(input_dir_list_str);
    // if (!dir_list_reader.is_open())
    // {
    //     return EXIT_FAILURE;
    // }
    // else
    // {
    //     std::string line;
    //     while (std::getline(dir_list_reader, line))
    //         input_dir_list.push_back(fs::path(line));
    //     dir_list_reader.close();
    // }
    
    // std::vector<std::string> prefixes;
    // boost::split(prefixes, prefixes_str, boost::is_any_of(","));

    // // fs::path input_dir (vm["input-dir"].as<std::string>());
    // std::vector<std::vector<std::pair<fs::path,fs::path> > > sequences;
    // for (fs::path input_dir : input_dir_list)
    // {
    //     std::vector<uls::Timestamp> log1 = uls::read_log_file(input_dir / fs::path("rs.log"));
    //     std::vector<uls::Timestamp> log2 = uls::read_log_file(input_dir / fs::path("pt.log"));
    //     std::vector<std::pair<uls::Timestamp,uls::Timestamp> > log_sync;
    //     uls::time_sync(log1, log2, log_sync);

    //     std::vector<std::pair<fs::path,fs::path> > s;
    //     for (int i = 0; i < log_sync.size(); i++)
    //     {
    //         prefixes[0] + log_sync[i].first.id + 
    //     }


    // }

    //     // std::vector<std::vector<fs::path> > s;
    //     // for (int i = 0; i < prefixes.size(); i++)
    //     // {
    //     //     std::vector<fs::path> frames_dir = uls::list_files_in_directory(input_dir / fs::path(prefixes[i]), vm["file-ext"].as<std::string>());
    //     //     std::sort(frames_dir.begin(), frames_dir.end());  // sort files by filename
    //     //     s.push_back(frames_dir);
    //     // }
    //     // sequences.push_back(s);
    // }

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

