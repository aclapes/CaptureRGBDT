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
#include <math.h>

#include "utils/common.hpp"
#include "utils/calibration.hpp"
#include "utils/synchronization.hpp"

bool debug = true;

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace cv;

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

template <typename T>
float distancePointLine(const cv::Point_<T> point, const cv::Vec<T,3>& line)
{
  //Line is given as a*x + b*y + c = 0
  return std::fabs(line(0)*point.x + line(1)*point.y + line(2))
      / std::sqrt(line(0)*line(0)+line(1)*line(1));
}
/**
 * \brief Compute and draw the epipolar lines in two images
 *      associated to each other by a fundamental matrix
 *
 * \param title     Title of the window to display
 * \param F         Fundamental matrix
 * \param img1      First image
 * \param img2      Second image
 * \param points1   Set of points in the first image
 * \param points2   Set of points in the second image matching to the first set
 * \param inlierDistance      Points with a high distance to the epipolar lines are
 *                not displayed. If it is negative, all points are displayed
 **/
template <typename T1, typename T2>
cv::Mat drawEpipolarLines(const cv::Matx<T1,3,3> F,
                const cv::Mat& img1, const cv::Mat& img2,
                const std::vector<cv::Point_<T2>> points1,
                const std::vector<cv::Point_<T2>> points2,
                const float inlierDistance = -1)
{
  CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
  cv::Mat outImg(img1.rows, img1.cols*2, CV_8UC3);
  cv::Rect rect1(0,0, img1.cols, img1.rows);
  cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);
  /*
   * Allow color drawing
   */
  if (img1.type() == CV_8U)
  {
    cv::cvtColor(img1, outImg(rect1), cv::COLOR_GRAY2BGR);
    cv::cvtColor(img2, outImg(rect2), cv::COLOR_GRAY2BGR);
  }
  else
  {
    img1.copyTo(outImg(rect1));
    img2.copyTo(outImg(rect2));
  }
  std::vector<cv::Vec<T2,3>> epilines1, epilines2;
  cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1
  cv::computeCorrespondEpilines(points2, 2, F, epilines2);
 
  CV_Assert(points1.size() == points2.size() &&
        points2.size() == epilines1.size() &&
        epilines1.size() == epilines2.size());
 
  cv::RNG rng(0);
  for(size_t i=0; i<points1.size(); i++)
  {
    if(inlierDistance > 0)
    {
      if(distancePointLine(points1[i], epilines2[i]) > inlierDistance ||
        distancePointLine(points2[i], epilines1[i]) > inlierDistance)
      {
        //The point match is no inlier
        continue;
      }
    }
    /*
     * Epipolar lines of the 1st point set are drawn in the 2nd image and vice-versa
     */
    cv::Scalar color(rng(256),rng(256),rng(256));
 
    cv::line(outImg(rect2),
      cv::Point(0,-epilines1[i][2]/epilines1[i][1]),
      cv::Point(img1.cols,-(epilines1[i][2]+epilines1[i][0]*img1.cols)/epilines1[i][1]),
      color);
    cv::circle(outImg(rect1), points1[i], 3, color, -1, cv::LINE_AA);
 
    cv::line(outImg(rect1),
      cv::Point(0,-epilines2[i][2]/epilines2[i][1]),
      cv::Point(img2.cols,-(epilines2[i][2]+epilines2[i][0]*img2.cols)/epilines2[i][1]),
      color);
    cv::circle(outImg(rect2), points2[i], 3, color, -1, cv::LINE_AA);
  }
  
  return outImg;
}

int main(int argc, char * argv[]) try
{
    /* -------------------------------- */
    /*   Command line argument parsing  */
    /* -------------------------------- */
    
    std::string corners_filepath_1, corners_filepath_2, intrinsics_filepath_1, intrinsics_filepath_2;
    std::string extrinsics_filepath;
    // bool vflip = false;
    bool verbose = false;

    po::options_description desc("Program options");
    desc.add_options()
        ("help,h", "Print help messages")
        ("corners_file-1", po::value<std::string>(&corners_filepath_1)->required(), "")
        ("corners_file-2", po::value<std::string>(&corners_filepath_2)->required(), "")
        ("intrinsics_file-1", po::value<std::string>(&intrinsics_filepath_1)->required(), "")
        ("intrinsics_file-2", po::value<std::string>(&intrinsics_filepath_2)->required(), "")
        ("extrinsics_file", po::value<std::string>(&extrinsics_filepath)->required(), "")
        ("intermediate-file,e", po::value<std::string>()->default_value(""), "Intermediate file")
        ("parent-dir", po::value<std::string>()->default_value(""), "Parent directory")
        // ("corners,c", po::value<std::string>()->default_value("./corners.yml"), "")
        // ("corner-selection,s", po::value<std::string>()->default_value("./corner-selection.yml"), "")
        // ("intrinsics,i", po::value<std::string>()->default_value("./intrinsics.yml"), "")
        // ("modality,m", po::value<std::string>()->default_value("thermal"), "Visual modality")
        // ("file-ext,x", po::value<std::string>()->default_value(".jpg"), "Image file extension")
        // ("verbose,v", po::bool_switch(&verbose), "Verbosity")
        ("nb-clusters,k", po::value<int>()->default_value(50), "Number of k-means clusters")
        // ("vflip,f", po::bool_switch(&vflip), "Vertical flip registered images")
        ("sync-delay", po::value<int>()->default_value(30), "Maximum time delay between RS and PT (in milliseconds)")
        // ("output-parameters,o", po::value<std::string>()->default_value(""), "Output parameters")
        ;

    
    po::positional_options_description positional_options; 
    positional_options.add("corners_file-1", 1); 
    positional_options.add("corners_file-2", 1); 
    positional_options.add("intrinsics_file-1", 1); 
    positional_options.add("intrinsics_file-2", 1); 
    positional_options.add("extrinsics_file", 1); 

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

    cv::FileStorage corners_fs_1 (corners_filepath_1, cv::FileStorage::READ);
    cv::FileStorage corners_fs_2 (corners_filepath_2, cv::FileStorage::READ);

    std::string prefix_1, prefix_2;
    corners_fs_1["prefix"] >> prefix_1;
    corners_fs_2["prefix"] >> prefix_2;

    std::string file_ext_1, file_ext_2;  // tbd
    corners_fs_1["file-extension"] >> file_ext_1;
    corners_fs_2["file-extension"] >> file_ext_2;

    // std::string modality_1, modality_2; // tbd
    // corners_fs_1["modality"] >> modality_1;
    // corners_fs_2["modality"] >> modality_2;

    int nb_sequences_1, nb_sequences_2;
    corners_fs_1["nb_sequences"] >> nb_sequences_1;
    corners_fs_2["nb_sequences"] >> nb_sequences_2;
    if (nb_sequences_1 != nb_sequences_2)
    {
        std::cerr << "Number of calibration sequences do not match: " 
                  << nb_sequences_1 << " and " << nb_sequences_2 << std::endl;
        return EXIT_FAILURE;
    }

    cv::Size pattern_size_1, pattern_size_2;
    corners_fs_1["pattern_size"] >> pattern_size_1;
    corners_fs_2["pattern_size"] >> pattern_size_2;
    if (((pattern_size_1.width - pattern_size_2.width) != 0 && abs(pattern_size_1.width - pattern_size_2.width) != 2)
      || ((pattern_size_1.height - pattern_size_2.height) != 0 && abs(pattern_size_1.height - pattern_size_2.height) != 2))
    {
        // see intersect_patterns(...)
        std::cerr << "Pattern sizes must be equal (or differ in exactly two squares), but they are: "
                  << pattern_size_1 << " and " << pattern_size_2 << std::endl;
        return EXIT_FAILURE;
    }

    std::string serial_number_1, serial_number_2;
    corners_fs_1["serial_number"] >> serial_number_1;
    corners_fs_2["serial_number"] >> serial_number_2;
    
    cv::Size resize_dims_1, resize_dims_2;
    corners_fs_1["resize_dims"] >> resize_dims_1;
    corners_fs_2["resize_dims"] >> resize_dims_2;
    if (resize_dims_1.width != resize_dims_2.width || resize_dims_1.height != resize_dims_2.height)
    {
        std::cerr << "Image dims in the two views need to match: " 
                  << resize_dims_1 << " and " << resize_dims_2 << std::endl;
        return EXIT_FAILURE;
    }

    // Read frames and corners

    std::vector<std::string> frames_all_1, frames_all_2;
    cv::Mat corners_all_1, corners_all_2;

    fs::path parent_path (vm["parent-dir"].as<std::string>());
    int sync_delay = vm["sync-delay"].as<int>();

    for (int i = 0; i < nb_sequences_1; i++)
    {
        std::string sequence_dir_1, sequence_dir_2;
        corners_fs_1["sequence_dir-" + std::to_string(i)] >> sequence_dir_1;
        corners_fs_2["sequence_dir-" + std::to_string(i)] >> sequence_dir_2;
        assert(sequence_dir_1 == sequence_dir_2);

        fs::path log_filepath_1 = (fs::path(sequence_dir_1) / prefix_1).string() + ".log";
        fs::path log_filepath_2 = (fs::path(sequence_dir_2) / prefix_2).string() + ".log";

        if (fs::path(sequence_dir_1).is_relative())
            log_filepath_1 = parent_path / log_filepath_1;
        if (fs::path(sequence_dir_2).is_relative())
            log_filepath_2 = parent_path / log_filepath_2;

        std::vector<uls::Timestamp> log_1 = uls::read_log(fs::path(sequence_dir_1) / fs::path(corners_fs_1["log-file"]));
        std::vector<uls::Timestamp> log_2 = uls::read_log(fs::path(sequence_dir_2) / fs::path(corners_fs_2["log-file"]));
        std::vector<std::pair<uls::Timestamp,uls::Timestamp> > log_12;
        uls::time_sync(uls::read_log(log_filepath_1), uls::read_log(log_filepath_2), log_12, sync_delay);

        // std::vector<int> indices_1, indices_2;
        // uls::time_sync(uls::read_log(log_filepath_1), uls::read_log(log_filepath_2), indices_1, indices_2, sync_delay);
        // assert(indices_1.size() == indices_2.size());

        std::vector<std::string> frames_1, frames_2;
        corners_fs_1["frames-" + std::to_string(i)] >> frames_1;
        corners_fs_2["frames-" + std::to_string(i)] >> frames_2;

        // for (int j = 0; j < frames_1.size(); j++)
        //     if (fs::path(frames_1[j]).is_relative) frames_1[j] = (parent_path / frames_1[j]).string();
        // for (int j = 0; j < frames_2.size(); j++)
        //     if (fs::path(frames_2[j]).is_relative) frames_2[j] = (parent_path / frames_2[j]).string();

        cv::Mat corners_1, corners_2;
        corners_fs_1["corners-" + std::to_string(i)] >> corners_1;
        corners_fs_2["corners-" + std::to_string(i)] >> corners_2;

        std::map<std::string, int> map_1, map_2;
        vector_to_map<std::string>(frames_1, map_1);
        vector_to_map<std::string>(frames_2, map_2);

        std::map<std::string, int>::iterator it_1, it_2;
        for (int j = 0; j < log_12.size(); j++)
        // for (int j = 0; j < indices_1.size(); j++)
        {
            fs::path frame_path_1 = fs::path(sequence_dir_1) / fs::path(prefix_1) / log_12[j].first.id;
            fs::path frame_path_2 = fs::path(sequence_dir_2) / fs::path(prefix_2) / log_12[j].second.id;

            // if (frame_path_1.is_relative())
            //     frame_path_1 = parent_path / frame_path_1;
            // if (frame_path_2.is_relative())
            //     frame_path_2 = parent_path / frame_path_2;

            it_1 = map_1.find(frame_path_1.string());
            it_2 = map_2.find(frame_path_2.string());
            if (it_1 != map_1.end() && it_2 != map_2.end())
            {
                std::string frame_with_corners_1 = frames_1[it_1->second];
                std::string frame_with_corners_2 = frames_2[it_2->second];

                if (fs::path(frame_with_corners_1).is_relative())
                    frame_with_corners_1 = (parent_path / frame_with_corners_1).string();
                if (fs::path(frame_with_corners_2).is_relative())
                    frame_with_corners_2 = (parent_path / frame_with_corners_2).string();

                frames_all_1.push_back(frame_with_corners_1);
                frames_all_2.push_back(frame_with_corners_2);

                // Re-orient corners so first corner is the top-left and last corner the bottom-right one
                corners_all_1.push_back( uls::orient_corners(corners_1.row(it_1->second)) );
                corners_all_2.push_back( uls::orient_corners(corners_2.row(it_2->second)) );
            }

            // std::string key_1 = frames_1[indices_1[j]];
            // std::string key_2 = frames_2[indices_2[j]];

            // it_1 = map_1.find(key_1);
            // it_2 = map_2.find(key_2);
            // if (it_1 != map_1.end() && it_2 != map_2.end())
            // {
            //     fs::path frame_path_1, frame_path_2;
            //     if (fs::path(frames_1[it_1->second]).is_relative())
            //         frame_path_1 = parent_path / fs::path(frames_1[it_1->second]);
            //     if (fs::path(frames_2[it_1->second]).is_relative())
            //         frame_path_2 = parent_path / fs::path(frames_2[it_2->second]);

            //     frames_all_1.push_back(frame_path_1.string());
            //     frames_all_2.push_back(frame_path_2.string());
                
            //     // Re-orient corners so first corner is the top-left and last corner the bottom-right one
            //     corners_all_1.push_back( uls::orient_corners(corners_1.row(it_1->second)) );
            //     corners_all_2.push_back( uls::orient_corners(corners_2.row(it_2->second)) );
            // }

            // fs::path frame_path_1, frame_path_2;
            // if (fs::path(frames_1[indices_1[j]]).is_relative())
            //     frame_path_1 = parent_path /);
            // if (fs::path(frames_2[indices_2[j]]).is_relative())
            //     frame_path_2 = parent_path / fs::path(frames_2[indices_2[j]]);

            // frames_all_1.push_back(frame_path_1.string());
            // frames_all_2.push_back(frame_path_2.string());
            // corners_all_1.push_back( uls::orient_corners(corners_1.row(indices_1[j])) );
            // corners_all_2.push_back( uls::orient_corners(corners_2.row(indices_2[j])) );
        }
    }

    assert(frames_all_1.size() == frames_all_2.size());
    assert(corners_all_1.rows == corners_all_2.rows);

    corners_fs_1.release();
    corners_fs_2.release();

    // Read intrinsics to use in the calibration of extrinsics

    cv::FileStorage intrinsics_fs_1, intrinsics_fs_2;
    cv::Mat camera_matrix_1, camera_matrix_2;
    cv::Mat dist_coeffs_1, dist_coeffs_2;
    cv::Point2f square_size_1, square_size_2;

    if (intrinsics_fs_1.open(intrinsics_filepath_1, cv::FileStorage::READ)
        && intrinsics_fs_2.open(intrinsics_filepath_2, cv::FileStorage::READ))
    {
        intrinsics_fs_1["camera_matrix"] >> camera_matrix_1;
        intrinsics_fs_2["camera_matrix"] >> camera_matrix_2;
        intrinsics_fs_1["dist_coeffs"] >> dist_coeffs_1;
        intrinsics_fs_2["dist_coeffs"] >> dist_coeffs_2;
        intrinsics_fs_1["square_size"] >> square_size_1;
        intrinsics_fs_2["square_size"] >> square_size_2;

        intrinsics_fs_1.release();
        intrinsics_fs_2.release();
    }

    // Select frames for extrinsic calibration

    cv::FileStorage intermediate_fs (vm["intermediate-file"].as<std::string>(), cv::FileStorage::READ);
    cv::Mat corners_selection_1, corners_selection_2;
    std::vector<std::string> frames_selection_1, frames_selection_2;
    cv::Size pattern_size;

    // Try opening intermediate file containing frames and corners from previous selection

    int K = vm["nb-clusters"].as<int>();

    if (intermediate_fs.isOpened())
    {
        intermediate_fs.open(vm["intermediate-file"].as<std::string>(), cv::FileStorage::READ);
        
        intermediate_fs["corners_selection-1"] >> corners_selection_1;
        intermediate_fs["corners_selection-2"] >> corners_selection_2;
        intermediate_fs["frames_selection-1"] >> frames_selection_1;
        intermediate_fs["frames_selection-2"] >> frames_selection_2;
        intermediate_fs["pattern_size"] >> pattern_size;
        intermediate_fs["nb-clusters"] >> K;
        intermediate_fs["sync_delay"] >> sync_delay;
        intermediate_fs.release();
    }
    else // New selection
    {
        // Cluster patterns corner positions so we ensure a proper coverage of the camera space

        cv::Mat labels, centers;
        cv::kmeans(corners_all_1, K, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

        // Shuffle the order of cluster elements

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
            
        /* Interactive selection procedure */

        std::vector<cv::Mat> corners_tmp_1 (K), corners_tmp_2 (K);
        std::vector<std::string> corner_frames_tmp_1 (K), corner_frames_tmp_2 (K);
        std::vector<int> ptr (K);
        
        int k = 0;
        while (true)
        {
            // Print current frame
            std::cout << k << ":" << ptr[k]+1 << "/" << indices[k].size() << std::endl;

            // Get current frame's shuffled index
            int idx = indices[k][ptr[k]];

            // Corners were detected in a different smaller/larger image resolution? Transform point domain space
            cv::Mat corners_row_1_transf, corners_row_2_transf;
            cv::Size frame_size_transf;
            uls::homogeneize_2d_domains(corners_all_1.row(idx), corners_all_2.row(idx), 
                                        resize_dims_1, resize_dims_2, 
                                        corners_row_1_transf, corners_row_2_transf, frame_size_transf);

            // Read images and associated corners
            cv::Mat img_1 = cv::imread(frames_all_1[idx], cv::IMREAD_UNCHANGED);
            cv::Mat img_2 = cv::imread(frames_all_2[idx], cv::IMREAD_UNCHANGED);
            
            // IF one the two patterns is smaller by an integer: (P1.width,P2.height) & (P2.width,P2.height),  
            // where P1.width == (P2.width - L) and P1.height ==  (P2.height - L). Get the intersection of both.
            cv::Mat corners_row_1_aligned, corners_row_2_aligned;
            cv::Size pattern_size_tmp;
            uls::intersect_patterns(corners_row_1_transf, corners_row_2_transf, 
                                    pattern_size_1, pattern_size_2, 
                                    corners_row_1_aligned, corners_row_2_aligned, pattern_size_tmp);

            pattern_size = pattern_size_tmp;
        
            // some modality might require preprocessing
            if (prefix_1 == "pt/thermal" )
                uls::thermal_to_8bit(img_1, img_1, cv::Rect(), cv::COLORMAP_BONE);
            if (prefix_2 == "pt/thermal")
                uls::thermal_to_8bit(img_2, img_2, cv::Rect(), cv::COLORMAP_BONE);

            // resize image to frame size where corners where detected when using calib-corners
            uls::resize(img_1, img_1, frame_size_transf);
            uls::resize(img_2, img_2, frame_size_transf);

            // you probably want to visualize the image + corners
            if (img_1.channels() == 1) // corners look better in a colorful image
                cv::cvtColor(img_1, img_1, cv::COLOR_GRAY2BGR);
            if (img_2.channels() == 1) // corners look better in a colorful image
                cv::cvtColor(img_2, img_2, cv::COLOR_GRAY2BGR);

            // draw the corners in tiled images
            std::vector<cv::Mat> tiling = {img_1, img_2};
            cv::drawChessboardCorners(tiling[0], pattern_size, corners_row_1_aligned, true);
            cv::drawChessboardCorners(tiling[1], pattern_size, corners_row_2_aligned, true);

            cv::Mat img;
            uls::tile(tiling, 640, 720, 1, 2, img);

            // draw a counter in the top-left part of the viewer with some information for the human annotator,
            // that is current cluster being selected + current element whithin that cluster

            std::stringstream ss;
            if (corner_frames_tmp_1[k] == frames_all_1[idx]) ss << "[*" << k << "*]";
            else ss << "[ " << k << " ]";
            ss << ' ' << ptr[k] << '/' << indices[k].size(); 
            cv::putText(img, ss.str(), cv::Point(img.cols/20.0,img.rows/20.0), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,0,255), 1, 8, false);

            // show image + corners to the annotator

            cv::imshow("Viewer", img);
            char ret = cv::waitKey();

            // Instructions for interaction with the viewer (same than those from calib-intrinsics):
            //
            // 'ENTER' to select the current viewed frame (cluster element) as its representative
            // ';' to move to next frame within that cluster
            // 'j' to move to previous frame
            // 'l' (or SPACE) to move to next cluster WITHOUT selecting the frame for that cluster
            // 'k' to move to previous cluster
            // 'ESC' to finish the interaction and move forward with the N selected frames, where N <= K
            //
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
                // ENTER
                if (corner_frames_tmp_1[k] == frames_all_1[idx])
                {
                    corner_frames_tmp_1[k] = corner_frames_tmp_2[k] = std::string();
                    corners_tmp_1[k] = corners_tmp_2[k] = cv::Mat();
                }
                else
                {
                    corners_tmp_1[k] = corners_row_1_aligned;// corners_all_1.row(idx);
                    corners_tmp_2[k] = corners_row_2_aligned;//corners_all_2.row(idx);
                    corner_frames_tmp_1[k] = frames_all_1[idx];
                    corner_frames_tmp_2[k] = frames_all_2[idx];
                    k = (k < (K - 1)) ? k + 1 : 0;
                }
            }
            else if (ret == 27)
                break;
        }

        assert(corner_frames_tmp_1.size() == corner_frames_tmp_2.size());

        // some irrelevant data re-organization 

        for (int k = 0; k < corner_frames_tmp_1.size(); k++)
        {
            if (!corner_frames_tmp_1[k].empty())
            {
                corners_selection_1.push_back(corners_tmp_1[k]);
                corners_selection_2.push_back(corners_tmp_2[k]);
                frames_selection_1.push_back(corner_frames_tmp_1[k]);
                frames_selection_2.push_back(corner_frames_tmp_2[k]);
            }
        }

        // save to intermediate file

        intermediate_fs.open(vm["intermediate-file"].as<std::string>(), cv::FileStorage::WRITE);
        intermediate_fs << "corners_selection-1" << corners_selection_1;
        intermediate_fs << "corners_selection-2" << corners_selection_2;
        intermediate_fs << "frames_selection-1" << frames_selection_1;
        intermediate_fs << "frames_selection-2" << frames_selection_2;
        intermediate_fs << "pattern_size" << pattern_size;
        intermediate_fs << "nb-clusters" << K;
        intermediate_fs << "sync_delay" << sync_delay;
        intermediate_fs.release();
    }

    // prepare some data structures

    assert(square_size_1.x == square_size_2.x && square_size_1.y == square_size_2.y);

    std::vector<std::vector<cv::Point2f> > image_points_1, image_points_2;
    uls::mat_to_vecvec<cv::Point2f>(corners_selection_1, image_points_1);
    uls::mat_to_vecvec<cv::Point2f>(corners_selection_2, image_points_2);

    std::vector<std::vector<cv::Point3f> > object_points (1);
    uls::calcBoardCornerPositions(pattern_size, square_size_1.x, square_size_1.y, object_points[0]);
    object_points.resize(image_points_1.size(), object_points[0]);
    cv::Mat dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);

    // eventually, calibrate!

    cv::Mat R, T, E, F;
    int flags = cv::CALIB_FIX_INTRINSIC;
    double rms = cv::stereoCalibrate(object_points,
                                     image_points_1, image_points_2, 
                                     camera_matrix_1, dist_coeffs_1, 
                                     camera_matrix_2, dist_coeffs_2,
                                     cv::Size(1280, 720),
                                     R, T, E, F,
                                     flags,
                                     cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 100, 1e-5));

    std::cout << "RMS error: " << rms << std::endl;

    // save calibration results

    cv::FileStorage extrinsics_fs (extrinsics_filepath, cv::FileStorage::WRITE);

    if (extrinsics_fs.isOpened())
    {
        extrinsics_fs << "num_modalities" << 2;

        extrinsics_fs << "modality-1" << prefix_1;
        extrinsics_fs << "modality-2" << prefix_2;

        extrinsics_fs << "serial_number-1" << serial_number_1;
        extrinsics_fs << "serial_number-2" << serial_number_2;

        extrinsics_fs << "camera_matrix-1" << camera_matrix_1;
        extrinsics_fs << "camera_matrix-2" << camera_matrix_2;
        extrinsics_fs << "dist_coeffs-1" << dist_coeffs_1;
        extrinsics_fs << "dist_coeffs-2" << dist_coeffs_2;

        extrinsics_fs << "R" << R;
        extrinsics_fs << "T" << T;
        // extrinsics_fs << "E" << E;
        // extrinsics_fs << "F" << F;

        extrinsics_fs << "resize_dims" << resize_dims_1;

        extrinsics_fs << "sync_delay" << sync_delay;
        extrinsics_fs << "flags" << flags;
        extrinsics_fs << "rms" << rms;
        extrinsics_fs << "original_file" << fs::basename(extrinsics_filepath);

        extrinsics_fs.release();
    }

    return EXIT_SUCCESS;
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

