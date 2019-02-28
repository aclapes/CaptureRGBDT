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

template<typename T>
void vector_to_map(std::vector<T> v, std::map<T, int> & m)
{
    m.clear();

    for (int i = 0; i < v.size(); i++)
    {
        m[v[i]] = i;
    }
}

int main(int argc, char * argv[]) try
{
    /* -------------------------------- */
    /*   Command line argument parsing  */
    /* -------------------------------- */
    
    std::string corners_file_1, corners_file_2, intrinsics_file_1, intrinsics_file_2;
    bool verbose = false;

    po::options_description desc("Program options");
    desc.add_options()
        ("help,h", "Print help messages")
        // ("corners,c", po::value<std::string>()->default_value("./corners.yml"), "")
        // ("corner-selection,s", po::value<std::string>()->default_value("./corner-selection.yml"), "")
        // ("intrinsics,i", po::value<std::string>()->default_value("./intrinsics.yml"), "")
        // ("modality,m", po::value<std::string>()->default_value("thermal"), "Visual modality")
        // ("file-ext,x", po::value<std::string>()->default_value(".jpg"), "Image file extension")
        // ("verbose,v", po::bool_switch(&verbose), "Verbosity")
        ("nb-clusters,k", po::value<int>()->default_value(50), "Number of k-means clusters")
        ("extrinsics-file,e", po::value<std::string>()->default_value(""), "Extrinsics")
        ("corners-1", po::value<std::string>(&corners_file_1)->required(), "")
        ("corners-2", po::value<std::string>(&corners_file_2)->required(), "")
        ("intrinsics-1", po::value<std::string>(&intrinsics_file_1)->required(), "")
        ("intrinsics-2", po::value<std::string>(&intrinsics_file_2)->required(), "");
    
    po::positional_options_description positional_options; 
    positional_options.add("corners-1", 1); 
    positional_options.add("corners-2", 1); 
    positional_options.add("intrinsics-1", 1); 
    positional_options.add("intrinsics-2", 1); 

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

    cv::FileStorage corners_fs_1 (corners_file_1, cv::FileStorage::READ);
    cv::FileStorage corners_fs_2 (corners_file_2, cv::FileStorage::READ);

    std::string prefix_1, prefix_2;
    corners_fs_1["prefix"] >> prefix_1;
    corners_fs_2["prefix"] >> prefix_2;

    std::string file_ext_1, file_ext_2;
    corners_fs_1["file-extension"] >> file_ext_1;
    corners_fs_2["file-extension"] >> file_ext_2;

    std::string modality_1, modality_2;
    corners_fs_1["modality"] >> modality_1;
    corners_fs_2["modality"] >> modality_2;

    cv::Size frame_size_1, frame_size_2;
    corners_fs_1["resize_dims"] >> frame_size_1;
    corners_fs_2["resize_dims"] >> frame_size_2;

    int nb_sequence_dirs_1, nb_sequence_dirs_2;
    corners_fs_1["sequence_dirs"] >> nb_sequence_dirs_1;
    corners_fs_2["sequence_dirs"] >> nb_sequence_dirs_2;
    assert(nb_sequence_dirs_1 == nb_sequence_dirs_2);

    cv::Size pattern_size_1, pattern_size_2;
    corners_fs_1["pattern_size"] >> pattern_size_1;
    corners_fs_2["pattern_size"] >> pattern_size_2;

    std::vector<std::string> frames_all_1, frames_all_2;
    cv::Mat corners_all_1, corners_all_2;

    std::vector<std::vector<std::pair<int,int> > > frames_indices_all;
    for (int i = 0; i < nb_sequence_dirs_1; i++)
    {
        std::string sequence_dir_1, sequence_dir_2;
        corners_fs_1["sequence_dir-" + std::to_string(i)] >> sequence_dir_1;
        corners_fs_2["sequence_dir-" + std::to_string(i)] >> sequence_dir_2;
        assert(sequence_dir_1 == sequence_dir_2);

        std::vector<uls::Timestamp> log_1 = uls::read_log_file(fs::path(sequence_dir_1) / fs::path(corners_fs_1["log-file"]));
        std::vector<uls::Timestamp> log_2 = uls::read_log_file(fs::path(sequence_dir_2) / fs::path(corners_fs_2["log-file"]));
        std::vector<std::pair<uls::Timestamp,uls::Timestamp> > log_12;
        uls::time_sync(log_1, log_2, log_12);

        std::vector<std::string> frames_1, frames_2;
        corners_fs_1["frames-" + std::to_string(i)] >> frames_1;
        corners_fs_2["frames-" + std::to_string(i)] >> frames_2;

        std::map<std::string, int> map_1, map_2;
        vector_to_map<std::string>(frames_1, map_1);
        vector_to_map<std::string>(frames_2, map_2);

        cv::Mat corners_1, corners_2;
        corners_fs_1["corners-" + std::to_string(i)] >> corners_1;
        corners_fs_2["corners-" + std::to_string(i)] >> corners_2;

        std::vector<std::string> frames_1_aux, frames_2_aux;
        std::vector<cv::Mat> corners_1_aux, corners_2_aux;

        std::map<std::string, int>::iterator it_1, it_2;
        for (int j = 0; j < log_12.size(); j++)
        {
            std::string frame_path_1 = prefix_1 + log_12[j].first.id  + file_ext_1;
            std::string frame_path_2 = prefix_2 + log_12[j].second.id + file_ext_2;

            it_1 = map_1.find(frame_path_1);
            it_2 = map_2.find(frame_path_2);
            if (it_1 != map_1.end() && it_2 != map_2.end())
            {
                frames_all_1.push_back((fs::path(sequence_dir_1) / fs::path(frames_1[it_1->second])).string());
                frames_all_2.push_back((fs::path(sequence_dir_2) / fs::path(frames_2[it_2->second])).string());
                // Re-orient corners so first corner is the top-left and last corner the bottom-right one
                corners_all_1.push_back( uls::orient_corners(corners_1.row(it_1->second)) );
                corners_all_2.push_back( uls::orient_corners(corners_2.row(it_2->second)) );
            }
        }
    }

    corners_fs_1.release();
    corners_fs_2.release();

    assert(frames_all_1.size() == frames_all_2.size());
    assert(corners_all_1.rows == corners_all_2.rows);

    int K = vm["nb-clusters"].as<int>();

    cv::Mat labels, centers;
    cv::kmeans(corners_all_1, K, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

    // ---

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
    
    std::vector<cv::Mat> corners_tmp_1, corners_tmp_2;
    corners_tmp_1.resize(K);
    corners_tmp_2.resize(K);

    std::vector<std::string> corner_frames_tmp_1, corner_frames_tmp_2;
    corner_frames_tmp_1.resize(K);
    corner_frames_tmp_2.resize(K);

    cv::Size pattern_size;

    // int k = 0;
    // std::vector<int> ptr (K);
    // bool keep_selecting = true;

    // while (keep_selecting)
    // {
    //     std::cout << k << ":" << ptr[k]+1 << "/" << indices[k].size() << std::endl;
    //     int idx = indices[k][ptr[k]];//indices_k.at<int>(i,0);

    //     cv::Mat corners_row_1_transf, corners_row_2_transf;
    //     cv::Size frame_size_transf;
    //     uls::transform_point_domains(corners_all_1.row(idx), corners_all_2.row(idx), 
    //                                  frame_size_1, frame_size_2, 
    //                                  corners_row_1_transf, corners_row_2_transf, frame_size_transf);

    //     cv::Mat img_1;
    //     if (modality_1 == "Color")
    //         img_1 = uls::ColorFrame(fs::path(frames_all_1[idx]), frame_size_transf).mat();
    //     else if (modality_1 == "Thermal")
    //         img_1 = uls::ThermalFrame(fs::path(frames_all_1[idx]), frame_size_transf).mat();

    //     cv::Mat img_2;
    //     if (modality_2 == "Color")
    //         img_2 = uls::ColorFrame(fs::path(frames_all_2[idx]), frame_size_transf).mat();
    //     else if (modality_2 == "Thermal")
    //         img_2 = uls::ThermalFrame(fs::path(frames_all_2[idx]), frame_size_transf).mat();

    //     cv::cvtColor(img_1, img_1, cv::COLOR_GRAY2BGR);
    //     cv::cvtColor(img_2, img_2, cv::COLOR_GRAY2BGR);

    //     cv::Mat corners_row_1_aligned, corners_row_2_aligned;
    //     // uls::align_pattern_corners(corners_all_1.row(idx), corners_all_2.row(idx), pattern_size_1, pattern_size_2, corners_row_1, corners_row_2, pattern_size);
    //     uls::align_pattern_corners(corners_row_1_transf, corners_row_2_transf, 
    //                                pattern_size_1, pattern_size_2, 
    //                                corners_row_1_aligned, corners_row_2_aligned, pattern_size);
    //     cv::drawChessboardCorners(img_1, pattern_size, corners_row_1_aligned, true);
    //     cv::drawChessboardCorners(img_2, pattern_size, corners_row_2_aligned, true);

    //     // cv::Mat corners_row_1minus2 = corners_row_1_aligned - corners_row_2_aligned;
    //     // std::cout << corners_row_1minus2 << std::endl;
    //     // cv::Point2f pi_1, pf_1, pi_2, pf_2;
    //     // pi_1 = corners_row_1_aligned.at<cv::Point2f>(0,0);
    //     // pi_2 = corners_row_2_aligned.at<cv::Point2f>(0,0);
    //     // pf_1 = corners_row_1_aligned.at<cv::Point2f>(0,corners_row_1_aligned.cols-1);
    //     // pf_2 = corners_row_2_aligned.at<cv::Point2f>(0,corners_row_1_aligned.cols-1);

    //     // std::cout << pi_1 << "->" << pf_1 << std::endl;
    //     // std::cout << pi_2 << "->" << pf_2 << std::endl;

    //     std::vector<cv::Mat> tiling = {img_1, img_2};
    //     cv::Mat img;
    //     uls::tile(tiling, 800, 900, 1, 2, img);

    //     std::stringstream ss;
    //     if (corner_frames_tmp_1[k] == frames_all_1[idx])
    //         ss << "[*" << k << "*]";
    //     else
    //         ss << "[ " << k << " ]";
    //     ss << ' ' << ptr[k] << '/' << indices[k].size(); 
    //     cv::putText(img, ss.str(), cv::Point(img.cols/20.0,img.rows/20.0), CV_FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,0,255), 1, 8, false);
    //     cv::imshow("Viewer", img);

    //     char ret = cv::waitKey();
    //     if (ret == 'j')
    //         ptr[k] = (ptr[k] > 0) ? ptr[k] - 1 : ptr[k];
    //     else if (ret == ';')
    //         ptr[k] = (ptr[k] < (indices[k].size() - 1)) ? ptr[k] + 1 : ptr[k];
    //     else if (ret == 'k')
    //     {
    //         k = (k > 0) ? k - 1 : K-1;
    //     }
    //     else if (ret == 'l' || ret == ' ')
    //         k = (k < (K - 1)) ? k + 1 : 0;
    //     else if (ret == 13) 
    //     {
    //         if (corner_frames_tmp_1[k] == frames_all_1[idx])
    //         {
    //             corner_frames_tmp_1[k] = corner_frames_tmp_2[k] = std::string();
    //             corners_tmp_1[k] = corners_tmp_2[k] = cv::Mat();
    //         }
    //         else
    //         {
    //             corners_tmp_1[k] = corners_row_1_aligned;// corners_all_1.row(idx);
    //             corners_tmp_2[k] = corners_row_2_aligned;//corners_all_2.row(idx);
    //             corner_frames_tmp_1[k] = frames_all_1[idx];
    //             corner_frames_tmp_2[k] = frames_all_2[idx];
    //             k = (k < (K - 1)) ? k + 1 : 0;
    //         }
    //     }
    //     else if (ret == 27)
    //         keep_selecting = false;
    // }

    // assert(corner_frames_tmp_1.size() == corner_frames_tmp_2.size());

    cv::Mat corners_selection_1, corners_selection_2;
    
    // for (int k = 0; k < corner_frames_tmp_1.size(); k++)
    // {
    //     if (!corner_frames_tmp_1[k].empty())
    //     {
    //         corners_selection_1.push_back(corners_tmp_1[k]);
    //         corners_selection_2.push_back(corners_tmp_2[k]);
    //     }
    // }

    // ---

    cv::FileStorage extrinsics_fs;

    // if (!vm["extrinsics-file"].as<std::string>().empty())
    // {
    //     extrinsics_fs.open(vm["extrinsics-file"].as<std::string>(), cv::FileStorage::WRITE);
    //     extrinsics_fs << "corners_selection_1" << corners_selection_1;
    //     extrinsics_fs << "corners_selection_2" << corners_selection_2;
    //     extrinsics_fs.release();
    // }

    if (!vm["extrinsics-file"].as<std::string>().empty())
    {
        extrinsics_fs.open(vm["extrinsics-file"].as<std::string>(), cv::FileStorage::READ);
        extrinsics_fs["corners_selection_1"] >> corners_selection_1;
        cv::flip(corners_selection_1, corners_selection_1, 1);
        extrinsics_fs["corners_selection_2"] >> corners_selection_2;
        cv::flip(corners_selection_2, corners_selection_2, 1);
        extrinsics_fs.release();
    }

    cv::FileStorage intrinsics_fs_1 (intrinsics_file_1, cv::FileStorage::READ);
    cv::FileStorage intrinsics_fs_2 (intrinsics_file_2, cv::FileStorage::READ);

    cv::Mat camera_matrix_1, camera_matrix_2;
    cv::Mat dist_coeffs_1, dist_coeffs_2;
    intrinsics_fs_1["camera_matrix"] >> camera_matrix_1;
    intrinsics_fs_2["camera_matrix"] >> camera_matrix_2;
    intrinsics_fs_1["dist_coeffs"] >> dist_coeffs_1;
    intrinsics_fs_2["dist_coeffs"] >> dist_coeffs_2;

    intrinsics_fs_1.release();
    intrinsics_fs_2.release();

    // camera_matrix_1 = cv::getOptimalNewCameraMatrix(camera_matrix_1, dist_coeffs_1, cv::Size(1280,720), -1, cv::Size(), (cv::Rect *)0, true);
    // camera_matrix_2 = cv::getOptimalNewCameraMatrix(camera_matrix_2, dist_coeffs_2, cv::Size(1280,720), -1, cv::Size(), (cv::Rect *)0, true);

    std::vector<std::vector<cv::Point2f> > image_points_1, image_points_2;
    uls::mat_to_vecvec<cv::Point2f>(corners_selection_1, image_points_1);
        uls::mat_to_vecvec<cv::Point2f>(corners_selection_2, image_points_2);
    // uls::mat_to_vecvec<cv::Point2f>(corners_selection_2.reshape(2, corners_selection_2.rows), image_points_2);

    std::vector<std::vector<cv::Point3f> > object_points (1);
    // uls::calcBoardCornerPositions(pattern_size, 0.05f, 0.05f, object_points[0]);
    uls::calcBoardCornerPositions(cv::Size(9,6), 0.05f, 0.05f, object_points[0]);
    object_points.resize(image_points_1.size(), object_points[0]);

    // std::cout << camera_matrix_2 << std::endl;
    // cv::Mat multiplier = (cv::Mat_<double>(3,3) <<  1.5,  0., 0.,
    //                                                 0., 1.5, 0., 
    //                                                 0.,  0., 1.);
    // camera_matrix_2 = multiplier * camera_matrix_2;
    // std::cout << camera_matrix_2 << std::endl;

    // camera_matrix_1 = cv::getOptimalNewCameraMatrix(camera_matrix_1, dist_coeffs_1, cv::Size(1280,720), 1, cv::Size(), (cv::Rect *)0,  true);
    // camera_matrix_2 = cv::getOptimalNewCameraMatrix(camera_matrix_2, dist_coeffs_2, cv::Size(1280,720), 1, cv::Size(), (cv::Rect *)0,  true);

    cv::Mat R, T, E, F;
    double rms = cv::stereoCalibrate(object_points,
                                     image_points_1, image_points_2, 
                                     camera_matrix_1, dist_coeffs_1, 
                                     camera_matrix_2, dist_coeffs_2,
                                     cv::Size(1280,720),
                                     R, T, E, F,
                                     cv::CALIB_FIX_INTRINSIC + cv::CALIB_USE_INTRINSIC_GUESS, cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 100, 1e-5));
    std::cout << rms << std::endl;

    // camera_matrix_1 = cv::getOptimalNewCameraMatrix(camera_matrix_1, dist_coeffs_1, cv::Size(1280,720), -1, cv::Size(), (cv::Rect *)0, true);
    // camera_matrix_2 = cv::getOptimalNewCameraMatrix(camera_matrix_2, dist_coeffs_2, cv::Size(1280,720), -1, cv::Size(), (cv::Rect *)0, true);



    cv::Mat R1,R2,P1,P2,Q;
    cv::stereoRectify(camera_matrix_1, dist_coeffs_1, camera_matrix_2, dist_coeffs_2, cv::Size(1280,720), R, T, R1, R2, P1, P2, Q, 0, -0.5);

    cv::Mat mapi1_1, mapi2_1, mapi1_2, mapi2_2;
    cv::initUndistortRectifyMap(camera_matrix_1, dist_coeffs_1, cv::Mat(), cv::Mat(), cv::Size(1280,720), CV_32FC1, mapi1_1, mapi2_1);
    cv::initUndistortRectifyMap(camera_matrix_2, dist_coeffs_2, cv::Mat(), cv::Mat(), cv::Size(1280,720), CV_32FC1, mapi1_2, mapi2_2);
    cv::Mat mape1_1, mape2_1, mape1_2, mape2_2;
    cv::initUndistortRectifyMap(camera_matrix_1, dist_coeffs_1, R1, P1, cv::Size(1280,720), CV_32FC1, mape1_1, mape2_1);
    cv::initUndistortRectifyMap(camera_matrix_2, dist_coeffs_2, R2, P2, cv::Size(1280,720), CV_32FC1, mape1_2, mape2_2);
    
    for (int i = 0; i < frames_all_1.size(); i++)
    {
        cv::Mat img_1 = uls::ColorFrame(fs::path(frames_all_1[i]), cv::Size(1280,720)).mat();
        cv::Mat img_2 = uls::ThermalFrame(fs::path(frames_all_2[i]), cv::Size(1280,720)).mat();        
    
        std::vector<cv::Mat> tiling (5);
        cv::cvtColor(img_1, tiling[0], cv::COLOR_GRAY2BGR);
        cv::cvtColor(img_2, tiling[1], cv::COLOR_GRAY2BGR);
        cv::multiply(tiling[0], cv::Scalar(0,0,1), tiling[0]);
        cv::multiply(tiling[1], cv::Scalar(0,1,0), tiling[1]);
        cv::addWeighted(tiling[0], 1, tiling[1], 1, 0.0, tiling[2]);
    
        cv::Mat tmp_1, tmp_2;

        cv::remap(img_1, tmp_1, mapi1_1, mapi2_1, cv::INTER_LINEAR);
        cv::remap(img_2, tmp_2, mapi1_2, mapi2_2, cv::INTER_LINEAR);
        cv::cvtColor(tmp_1, tmp_1, cv::COLOR_GRAY2BGR);
        cv::cvtColor(tmp_2, tmp_2, cv::COLOR_GRAY2BGR);
        cv::multiply(tmp_1, cv::Scalar(0,0,1), tmp_1);
        cv::multiply(tmp_2, cv::Scalar(0,1,0), tmp_2);
        cv::addWeighted(tmp_1, 1, tmp_2, 1, 0.0, tiling[3]);

        cv::remap(img_1, tmp_1, mape1_1, mape2_1, cv::INTER_LINEAR);
        cv::remap(img_2, tmp_2, mape1_2, mape2_2, cv::INTER_LINEAR);
        cv::cvtColor(tmp_1, tmp_1, cv::COLOR_GRAY2BGR);
        cv::cvtColor(tmp_2, tmp_2, cv::COLOR_GRAY2BGR);
        cv::multiply(tmp_1, cv::Scalar(0,0,1), tmp_1);
        cv::multiply(tmp_2, cv::Scalar(0,1,0), tmp_2);
        cv::addWeighted(tmp_1, 1, tmp_2, 1, 0.0, tiling[4]);

        cv::Mat viewer_img;
        uls::tile(tiling, 960, 810, 2, 3, viewer_img);
        // cv::imshow("Viewer", tile);
        cv::imshow("Viewer", viewer_img);
        cv::waitKey(16);
    }

    /*
     * Define calibration pattern size
     */

    // std::vector<std::string> pattern_dims;
    // boost::split(pattern_dims, vm["pattern"].as<std::string>(), boost::is_any_of(","));
    // assert(pattern_dims.size() == 2);
    
    // int x = std::stoi(pattern_dims[0]);
    // int y = std::stoi(pattern_dims[1]);
    // assert(x > 2 && x > 2);

    // cv::Size pattern_size (x,y);
    
    
    // for (fs::path p : frames)
    // {
    //     cv::Mat img = uls::ThermalFrame(p).mat();
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

