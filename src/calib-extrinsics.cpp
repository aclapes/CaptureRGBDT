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
    
    std::string corners_file_1, corners_file_2, intrinsics_file_1, intrinsics_file_2;
    bool vflip = false;
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
        ("vflip,f", po::bool_switch(&vflip), "Vertical flip registered images")
        ("sync-delay", po::value<int>()->default_value(30), "Maximum time delay between RS and PT (in milliseconds)")
        ("intermediate-file,e", po::value<std::string>()->default_value(""), "Intermediate file")
        ("output-parameters,o", po::value<std::string>()->default_value(""), "Output parameters")
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

    // int y_shift_1, y_shift_2;
    // corners_fs_1["y-shift"] >> y_shift_1;
    // corners_fs_2["y-shift"] >> y_shift_2;

    // Read frames and corners

    std::vector<std::string> frames_all_1, frames_all_2;
    cv::Mat corners_all_1, corners_all_2;

    for (int i = 0; i < nb_sequence_dirs_1; i++)
    {
        std::string sequence_dir_1, sequence_dir_2;
        corners_fs_1["sequence_dir-" + std::to_string(i)] >> sequence_dir_1;
        corners_fs_2["sequence_dir-" + std::to_string(i)] >> sequence_dir_2;
        assert(sequence_dir_1 == sequence_dir_2);

        std::vector<uls::Timestamp> log_1 = uls::read_log_file(fs::path(sequence_dir_1) / fs::path(corners_fs_1["log-file"]));
        std::vector<uls::Timestamp> log_2 = uls::read_log_file(fs::path(sequence_dir_2) / fs::path(corners_fs_2["log-file"]));
        std::vector<std::pair<uls::Timestamp,uls::Timestamp> > log_12;
        uls::time_sync(log_1, log_2, log_12, vm["sync-delay"].as<int>());

        std::vector<std::string> frames_1, frames_2;
        corners_fs_1["frames-" + std::to_string(i)] >> frames_1;
        corners_fs_2["frames-" + std::to_string(i)] >> frames_2;

        cv::Mat corners_1, corners_2;
        corners_fs_1["corners-" + std::to_string(i)] >> corners_1;
        corners_fs_2["corners-" + std::to_string(i)] >> corners_2;

        std::map<std::string, int> map_1, map_2;
        vector_to_map<std::string>(frames_1, map_1);
        vector_to_map<std::string>(frames_2, map_2);

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

    assert(frames_all_1.size() == frames_all_2.size());
    assert(corners_all_1.rows == corners_all_2.rows);

    corners_fs_1.release();
    corners_fs_2.release();

    // Select frames for extrinsic calibration

    cv::FileStorage extrinsics_fs (vm["intermediate-file"].as<std::string>(), cv::FileStorage::READ);
    cv::Mat corners_selection_1, corners_selection_2;
    std::vector<std::string> frames_selection_1, frames_selection_2;
    cv::Size pattern_size;

    // Try opening intermediate file containing frames and corners from previous selection
    if (extrinsics_fs.isOpened())
    {
        extrinsics_fs.open(vm["intermediate-file"].as<std::string>(), cv::FileStorage::READ);
        extrinsics_fs["pattern_size"] >> pattern_size;
        extrinsics_fs["corners_selection_1"] >> corners_selection_1;
        extrinsics_fs["corners_selection_2"] >> corners_selection_2;
        extrinsics_fs["frames_selection_1"] >> frames_selection_1;
        extrinsics_fs["frames_selection_2"] >> frames_selection_2;
        extrinsics_fs.release();
    }
    else // New selection
    {
        // Cluster patterns corner positions so we ensure a proper coverage of the camera space

        int K = vm["nb-clusters"].as<int>();
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
            
        // Interactive selection procedure

        std::vector<cv::Mat> corners_tmp_1 (K), corners_tmp_2 (K);
        std::vector<std::string> corner_frames_tmp_1 (K), corner_frames_tmp_2 (K);
        std::vector<int> ptr (K);
        
        int k = 0;
        while (true)
        {
            // Print current frame
            std::cout << k << ":" << ptr[k]+1 << "/" << indices[k].size() << std::endl;

            // Get current frame's shuffled index
            int idx = indices[k][ptr[k]]; //indices_k.at<int>(i,0);

            // Corners were detected in a different smaller/larger image resolution? Transform point domain space
            cv::Mat corners_row_1_transf, corners_row_2_transf;
            cv::Size frame_size_transf;
            uls::homogeneize_2d_domains(corners_all_1.row(idx), corners_all_2.row(idx), 
                                        frame_size_1, frame_size_2, 
                                        corners_row_1_transf, corners_row_2_transf, frame_size_transf);

            // Read images
            //cv::Mat img_1;
            //if (modality_1 == "Color")
            //    img_1 = uls::ColorFrame(fs::path(frames_all_1[idx]), frame_size_transf).get();
            //else if (modality_1 == "Thermal")
            //    img_1 = uls::ThermalFrame(fs::path(frames_all_1[idx]), frame_size_transf).get();

            //cv::Mat img_2;
            //if (modality_2 == "Color")
            //    img_2 = uls::ColorFrame(fs::path(frames_all_2[idx]), frame_size_transf).get();
            //else if (modality_2 == "Thermal")
            //    img_2 = uls::ThermalFrame(fs::path(frames_all_2[idx]), frame_size_transf).get();
            cv::Mat img_1 = cv::imread(frames_all_1[idx], cv::IMREAD_UNCHANGED);
            cv::Mat img_2 = cv::imread(frames_all_2[idx], cv::IMREAD_UNCHANGED);
            
            if (modality_1 == "pt/thermal" )
                uls::thermal_to_8bit(img_1, img_1, cv::Rect(), cv::COLORMAP_BONE);
            else if (modality_2 == "pt/thermal")
                uls::thermal_to_8bit(img_2, img_2, cv::Rect(), cv::COLORMAP_BONE);

            uls::resize(img_1, img_1, frame_size_transf);
            uls::resize(img_2, img_2, frame_size_transf);

            // cv::cvtColor(img_1, img_1, cv::COLOR_GRAY2BGR);
            // cv::cvtColor(img_2, img_2, cv::COLOR_GRAY2BGR);

            // IF one the two patterns is smaller by an integer: (P1.width,P2.height) & (P2.width,P2.height),  
            // where P1.width == (P2.width - L) and P1.height ==  (P2.height - L). Get the intersection of both.
            cv::Mat corners_row_1_aligned, corners_row_2_aligned;
            cv::Size pattern_size_tmp;
            uls::intersect_patterns(corners_row_1_transf, corners_row_2_transf, 
                                    pattern_size_1, pattern_size_2, 
                                    corners_row_1_aligned, corners_row_2_aligned, pattern_size_tmp);

        //   assert (pattern_size_tmp.width == pattern_size.width && pattern_size_tmp.height == pattern_size.height);
            pattern_size = pattern_size_tmp;

            // Compose the images, text, and interactions with the viewer

            std::vector<cv::Mat> tiling = {img_1, img_2};
            cv::drawChessboardCorners(tiling[0], pattern_size, corners_row_1_aligned, true);
            cv::drawChessboardCorners(tiling[1], pattern_size, corners_row_2_aligned, true);

            cv::Mat img;
            uls::tile(tiling, 800, 900, 1, 2, img);

            std::stringstream ss;
            if (corner_frames_tmp_1[k] == frames_all_1[idx]) ss << "[*" << k << "*]";
            else ss << "[ " << k << " ]";
            ss << ' ' << ptr[k] << '/' << indices[k].size(); 
            cv::putText(img, ss.str(), cv::Point(img.cols/20.0,img.rows/20.0), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,0,255), 1, 8, false);

            // Show viewer

            cv::imshow("Viewer", img);
            char ret = cv::waitKey();

            //
            // Interact with viewer. Controls:
            // 
            //  "j": go back one frame within that cluster
            //  ";": go forwared one frame within that same cluster
            //  "k": move to previous cluster
            //  "l"/SPACE: move to next one cluster
            //  ENTER: select that frame as representative for the current cluster (marked with asterisks in the top-left text in the Viewer) and move to next cluster.
            //         As if "k" was pressed. Important: if the frame was already selected, it removes the selection.
            //  ESC: finish selecting frames (ideally, you must chose as much as possible)
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

        // Process and save to intermediate file

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

        extrinsics_fs.open(vm["intermediate-file"].as<std::string>(), cv::FileStorage::WRITE);
        extrinsics_fs << "pattern_size" << pattern_size;
        extrinsics_fs << "corners_selection_1" << corners_selection_1;
        extrinsics_fs << "corners_selection_2" << corners_selection_2;
        extrinsics_fs << "frames_selection_1" << frames_selection_1;
        extrinsics_fs << "frames_selection_2" << frames_selection_2;
        extrinsics_fs << "sync_delay" << vm["sync-delay"].as<int>();
        extrinsics_fs.release();
    }

    // Read intrinsics to use in the calibration of extrinsics

    cv::FileStorage intrinsics_fs_1, intrinsics_fs_2;
    cv::Mat camera_matrix_1, camera_matrix_2;
    cv::Mat dist_coeffs_1, dist_coeffs_2;
    cv::Point2f square_size_1, square_size_2;

    if (intrinsics_fs_1.open(intrinsics_file_1, cv::FileStorage::READ)
        && intrinsics_fs_2.open(intrinsics_file_2, cv::FileStorage::READ))
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

    assert(square_size_1.x == square_size_2.x && square_size_1.y == square_size_2.y);

    // for (int i = 0; i < corners_selection_1.rows; i++)
    // {
    //     cv::undistortPoints(corners_selection_1.row(i), corners_selection_1.row(i), camera_matrix_1, dist_coeffs_1, cv::Mat(), camera_matrix_1);
    //     cv::undistortPoints(corners_selection_2.row(i), corners_selection_2.row(i), camera_matrix_2, dist_coeffs_2, cv::Mat(), camera_matrix_);
    // }

    std::vector<std::vector<cv::Point2f> > image_points_1, image_points_2;
    uls::mat_to_vecvec<cv::Point2f>(corners_selection_1, image_points_1);
    uls::mat_to_vecvec<cv::Point2f>(corners_selection_2, image_points_2);

    std::vector<std::vector<cv::Point3f> > object_points (1);
    uls::calcBoardCornerPositions(pattern_size, square_size_1.x, square_size_1.y, object_points[0]);
    object_points.resize(image_points_1.size(), object_points[0]);
    cv::Mat dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
    // Perform extrinsic calibration

    cv::Mat R, T, E, F;
    // int flags = CV_CALIB_USE_INTRINSIC_GUESS + 
    //             CV_CALIB_FIX_ASPECT_RATIO + 
    //             CV_CALIB_FIX_FOCAL_LENGTH + 
    //             CV_CALIB_FIX_K1 + 
    //             CV_CALIB_FIX_K2 + 
    //             CV_CALIB_FIX_K3 + 
    //             CV_CALIB_FIX_K4 +
    //             CV_CALIB_FIX_K5 + 
    //             CV_CALIB_FIX_K6;// + CV_CALIB_TILTED_MODEL;
    int flags = cv::CALIB_FIX_INTRINSIC;

    double rms = cv::stereoCalibrate(object_points,
                                     image_points_1, image_points_2, 
                                     camera_matrix_1, dist_coeffs_1, 
                                     camera_matrix_2, dist_coeffs_2,
                                     cv::Size(1280, 720),
                                     R, T, E, F,
                                     flags,
                                     cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 100, 1e-5));

    std::cout << rms << std::endl;

    if (!vm["output-parameters"].as<std::string>().empty())
    {
        extrinsics_fs.open(vm["output-parameters"].as<std::string>(), cv::FileStorage::WRITE);

        extrinsics_fs << "modality-1" << modality_1;
        extrinsics_fs << "modality-2" << modality_2;

        extrinsics_fs << "camera_matrix-1" << camera_matrix_1;
        extrinsics_fs << "camera_matrix-2" << camera_matrix_2;
        extrinsics_fs << "dist_coeffs-1" << dist_coeffs_1;
        extrinsics_fs << "dist_coeffs-2" << dist_coeffs_2;

        extrinsics_fs << "R" << R;
        extrinsics_fs << "T" << T;
        extrinsics_fs << "E" << E;
        extrinsics_fs << "F" << F;

        extrinsics_fs << "sync_delay" << vm["sync-delay"].as<int>();
        extrinsics_fs << "flags" << flags;
        extrinsics_fs << "rms" << rms;

        extrinsics_fs.release();
    }

    // Rectification

    // T.at<double>(1,0) = 0; // assume both cameras are completely horizontal in the stereo rig by removing offset along y-axis (rectification can compensate)

    // cv::Mat R1,R2,P1,P2,Q;
    // cv::stereoRectify(camera_matrix_1, dist_coeffs_1, camera_matrix_2, dist_coeffs_2, cv::Size(1280,720+abs(y_shift_1)), R, T, R1, R2, P1, P2, Q, 0);//, cv::Size(1.2*1280,1.2*(720+abs(y_shift_1))), &r1, &r2);
    // cv::Mat R1z,R2z,P1z,P2z,Qz;
    // cv::stereoRectify(camera_matrix_1, dist_coeffs_1, camera_matrix_2, dist_coeffs_2, cv::Size(1280,720+abs(y_shift_2)), R, T, R1z, R2z, P1z, P2z, Qz, cv::CALIB_ZERO_DISPARITY);

    // //
    // // Find inverse mappings that are used to convert original images to calibrated ones (using cv::remap). Three versions based on:
    // //
    // // (1) Only intrinsics
    // // (2) Intrinsics + Extrinsics (disparity: alignmed in the covered pattern space)
    // // (3) Intrinsics + Extrinsics (zero disparity: alignment at infinity)

    // cv::Mat imapx_1, imapy_1, imapx_2, imapy_2;
    // cv::initUndistortRectifyMap(camera_matrix_1, dist_coeffs_1, cv::Mat(), cv::Mat(), cv::Size(1280,720+abs(y_shift_1)), CV_32FC1, imapx_1, imapy_1);
    // cv::initUndistortRectifyMap(camera_matrix_2, dist_coeffs_2, cv::Mat(), cv::Mat(), cv::Size(1280,720+abs(y_shift_2)), CV_32FC1, imapx_2, imapy_2);
    
    // cv::Mat mapx_1, mapy_1, mapx_2, mapy_2;
    // cv::initUndistortRectifyMap(camera_matrix_1, dist_coeffs_1, R1, P1, cv::Size(1280,720+abs(y_shift_1)), CV_32FC1, mapx_1, mapy_1);
    // cv::initUndistortRectifyMap(camera_matrix_2, dist_coeffs_2, R2, P2, cv::Size(1280,720+abs(y_shift_2)), CV_32FC1, mapx_2, mapy_2);

    // cv::Mat mapxz_1, mapyz_1, mapxz_2, mapyz_2;
    // cv::initUndistortRectifyMap(camera_matrix_1, dist_coeffs_1, R1z, P1z, cv::Size(1280,720+abs(y_shift_1)), CV_32FC1, mapxz_1, mapyz_1);
    // cv::initUndistortRectifyMap(camera_matrix_2, dist_coeffs_2, R2z, P2z, cv::Size(1280,720+abs(y_shift_2)), CV_32FC1, mapxz_2, mapyz_2);

    // // Visualize calibrations

    // int grid_x = 3;
    // int grid_y = 2;

    // for (int i = 0; i < frames_all_1.size(); i++)
    // {
    //     cv::Mat img_1 = uls::ColorFrame(fs::path(frames_all_1[i]), cv::Size(1280,720), y_shift_1).mat();
    //     cv::Mat img_2 = uls::ThermalFrame(fs::path(frames_all_2[i]), cv::Size(1280,720), y_shift_2).mat();
    
    //     std::vector<cv::Mat> tiling (grid_x * grid_y);
    //     cv::cvtColor(img_1, tiling[0], cv::COLOR_GRAY2BGR);
    //     cv::cvtColor(img_2, tiling[1], cv::COLOR_GRAY2BGR);
    //     cv::multiply(tiling[0], cv::Scalar(0,0,1), tiling[0]);
    //     cv::multiply(tiling[1], cv::Scalar(0,1,0), tiling[1]);
    //     cv::addWeighted(tiling[0], 1, tiling[1], 1, 0.0, tiling[2]);
    
    //     cv::Mat tmp_1, tmp_2, tmp_r;

    //     cv::remap(img_1, tmp_1, imapx_1, imapy_1, cv::INTER_LINEAR);
    //     cv::remap(img_2, tmp_2, imapx_2, imapy_2, cv::INTER_LINEAR);
    //     cv::cvtColor(tmp_1, tmp_1, cv::COLOR_GRAY2BGR);
    //     cv::cvtColor(tmp_2, tmp_2, cv::COLOR_GRAY2BGR);
    //     cv::multiply(tmp_1, cv::Scalar(0,0,1), tmp_1);
    //     cv::multiply(tmp_2, cv::Scalar(0,1,0), tmp_2);
    //     cv::addWeighted(tmp_1, 1, tmp_2, 1, 0.0, tiling[3]);

    //     cv::remap(img_1, tmp_1, mapx_1, mapy_1, cv::INTER_LINEAR);
    //     cv::remap(img_2, tmp_2, mapx_2, mapy_2, cv::INTER_LINEAR);
    //     cv::cvtColor(tmp_1, tmp_1, cv::COLOR_GRAY2BGR);
    //     cv::cvtColor(tmp_2, tmp_2, cv::COLOR_GRAY2BGR);
    //     cv::multiply(tmp_1, cv::Scalar(0,0,1), tmp_1);
    //     cv::multiply(tmp_2, cv::Scalar(0,1,0), tmp_2);
    //     cv::addWeighted(tmp_1, 1, tmp_2, 1, 0.0, tmp_r);
    //     if (!vflip) tiling[4] = tmp_r;
    //     else cv::flip(tmp_r, tiling[4], 0);

    //     cv::remap(img_1, tmp_1, mapxz_1, mapyz_1, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(127,127,127));
    //     cv::remap(img_2, tmp_2, mapxz_2, mapyz_2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(127,127,127));
    //     cv::cvtColor(tmp_1, tmp_1, cv::COLOR_GRAY2BGR);
    //     cv::cvtColor(tmp_2, tmp_2, cv::COLOR_GRAY2BGR);
    //     cv::multiply(tmp_1, cv::Scalar(0,0,1), tmp_1);
    //     cv::multiply(tmp_2, cv::Scalar(0,1,0), tmp_2);
    //     cv::addWeighted(tmp_1, 1, tmp_2, 1, 0.0, tmp_r);
    //     if (!vflip) tiling[5] = tmp_r;
    //     else cv::flip(tmp_r, tiling[5], 0);

    //     cv::Mat viewer_img;
    //     uls::tile(tiling, 1920, 640, grid_x, grid_y, viewer_img);
    //     cv::imshow("Viewer", viewer_img);
    //     char ret = cv::waitKey(33); 
    //     if (ret == 13)
    //       break;
    // }

    // if (!vm["output-parameters"].as<std::string>().empty())
    // {
    //     extrinsics_fs.open(vm["output-parameters"].as<std::string>(), cv::FileStorage::WRITE);

    //     extrinsics_fs << "modality-1" << modality_1;
    //     extrinsics_fs << "modality-2" << modality_2;

    //     extrinsics_fs << "imapx-1" << imapx_1;
    //     extrinsics_fs << "imapy-1" << imapy_1;
    //     extrinsics_fs << "imapx-2" << imapx_2;
    //     extrinsics_fs << "imapy-2" << imapy_2;

    //     extrinsics_fs << "mapx-1" << mapx_1;
    //     extrinsics_fs << "mapy-1" << mapy_1;
    //     extrinsics_fs << "mapx-2" << mapx_2;
    //     extrinsics_fs << "mapy-2" << mapy_2;

    //     extrinsics_fs << "mapxz-1" << mapxz_1;
    //     extrinsics_fs << "mapyz-1" << mapyz_1;
    //     extrinsics_fs << "mapxz-2" << mapxz_2;
    //     extrinsics_fs << "mapyz-2" << mapyz_2;

    //     extrinsics_fs << "frame_size" << cv::Size(1280, 720);
    //     extrinsics_fs << "y-shift-1" << y_shift_1;
    //     extrinsics_fs << "y-shift-2" << y_shift_2;
    //     extrinsics_fs << "vflip" << vflip;

    //     extrinsics_fs.release();
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

