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
    
    std::string corners_filepath;
    std::string intrinsics_filepath;
    // std::string input_dir_list_str;
    bool verbose = false;

    po::options_description desc("Program options");
    desc.add_options()
        ("help,h", "Print help messages")
        ("corners-file", po::value<std::string>(&corners_filepath)->required(), "Input corners file")
        ("intrinsics-file", po::value<std::string>(&intrinsics_filepath)->required(), "Output file containing intrinsics")
        ("parent-dir", po::value<std::string>()->default_value(""), "Path containing the sequences")
        ("square-size,q", po::value<std::string>()->default_value("0.05,0.05"), "Square size in meters")
        ("nb-clusters,k", po::value<int>()->default_value(50), "Number of k-means clusters");

    po::positional_options_description positional_options; 
    positional_options.add("corners-file", 1);
    positional_options.add("intrinsics-file", 1); 

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

    // load some auxiliary variables from the corners file

    cv::FileStorage corners_fs (corners_filepath, cv::FileStorage::READ);
    if (!corners_fs.isOpened())
    {
        std::cerr << corners_filepath << " corners file not found." << std::endl;
        return EXIT_FAILURE;
    }

    std::string prefix;
    corners_fs["prefix"] >> prefix;
    boost::trim_if(prefix, &uls::is_bar);  // remove leading and ending '/' and '\' bars

    std::string serial_number; // sensor's serial number
    corners_fs["serial_number"] >> serial_number;

    int nb_sequence_dirs;
    corners_fs["nb_sequences"] >> nb_sequence_dirs;

    cv::Size frame_size;
    corners_fs["resize_dims"] >> frame_size;

    cv::Size pattern_size;
    corners_fs["pattern_size"] >> pattern_size;

    std::vector<std::string> square_size_aux;
    boost::split(square_size_aux, vm["square-size"].as<std::string>(), boost::is_any_of(","));

    assert(square_size_aux.size() == 1 || square_size_aux.size() == 2);
    float square_width  = std::stof(square_size_aux[0]);
    float square_height = std::stof(square_size_aux[square_size_aux.size()-1]); // obscure hack

    assert(square_width > 0 && square_height > 0);
    cv::Point2f square_size (square_width, square_height);
    
    // load the corners themselves

    std::vector<std::string> frames_all;
    for (int i = 0; i < nb_sequence_dirs; i++)
    {
        std::string sequence_dir;
        corners_fs["sequence_dir-" + std::to_string(i)] >> sequence_dir;

        std::vector<std::string> frames;
        corners_fs["frames-" + std::to_string(i)] >> frames;
        for (int j = 0; j < frames.size(); j++)
        {
            if (fs::path(frames[j]).is_relative())
                frames[j] = (fs::path(vm["parent-dir"].as<std::string>()) / fs::path(frames[j])).string();
            
            if (!fs::exists(frames[j]))
            {
                std::cerr << frames[j] << " not found" << std::endl;
                return EXIT_FAILURE;
            }
        }
            
        frames_all.insert(frames_all.end(), frames.begin(), frames.end());
    }

    // build a big matrix where each detected pattern's corners is a row we'll use in clustering

    cv::Mat corners_all (frames_all.size(), pattern_size.height * pattern_size.width * 2, CV_32FC1);
    int i, count;
    for (i = 0, count = 0; i < nb_sequence_dirs; i++)
    {
        cv::Mat corners_aux;
        corners_fs["corners-" + std::to_string(i)] >> corners_aux;
        cv::Mat c = corners_aux.reshape(1, corners_aux.rows);
        std::cout << c.rows << "," << c.cols << "," << c.channels() << std::endl;
        if (c.rows > 0)
            c.copyTo(corners_all(cv::Rect(0, count, pattern_size.height * pattern_size.width * 2, corners_aux.rows)));
        count += corners_aux.rows;
}

    corners_fs.release(); // we won't need this anymore

    // we will cluster the detected patterns to ensure where the K selected corners cover as much
    // image space as possible

    int K = vm["nb-clusters"].as<int>();

    cv::Mat labels, centers;
    cv::kmeans(corners_all, K, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);
 
    // convert the vector of assigned labels to a list of list of centroid assignations
    // so we can retrieve all the detected patterns assigned to a particular centroid

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
    
    // manual interactive selection of a detected pattern from each cluster

    std::vector<cv::Mat> corners (K);
    std::vector<std::string> corner_frames (K);
    int k = 0;
    std::vector<int> ptr (K);
    bool keep_selecting = true;

    cv::namedWindow("Viewer");

    while (keep_selecting)
    {
        // Print current frame
        std::cout << k << ":" << ptr[k]+1 << "/" << indices[k].size() << std::endl;

        // Get current frame's shuffled index
        int idx = indices[k][ptr[k]];
        
        // retrieve the image and associated corners
        cv::Mat img = cv::imread(frames_all[idx], cv::IMREAD_UNCHANGED);
        cv::Mat corners_aux = corners_all.row(idx).reshape(2, pattern_size.width * pattern_size.height);

        // the modality might require preprocessing
        if (prefix == "pt/thermal")
            uls::thermal_to_8bit(img, img); // e.g., thermal requires normalization and conversion from 16u to 8u

        // resize image to frame size where corners where detected when using calib-corners
        uls::resize(img, img, frame_size);

        // you probably want to visualize the image + corners
        if (img.channels() == 1) // corners look better in a colorful image
            cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

        // draw the corners
        cv::drawChessboardCorners(img, pattern_size, corners_aux, true);

        // draw a counter in the top-left part of the viewer with some information for the human annotator,
        // that is current cluster being selected + current element whithin that cluster
        std::stringstream ss;
        if (corner_frames[k] == frames_all[idx])
            ss << "[*" << k << "*]";
        else
            ss << "[ " << k << " ]";
        ss << ' ' << ptr[k] << '/' << indices[k].size(); 
        cv::putText(img, ss.str(), cv::Point(frame_size.width/20.0,frame_size.height/10.0), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,0,255), 1, 8, false);
        
        // show image + corners to the annotator

        cv::imshow("Viewer", img);
        char ret = cv::waitKey();

        // Instructions for interaction with the viewer:
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
        else if (ret == ';') // ';' move to next cluster element (frame)
            ptr[k] = (ptr[k] < (indices[k].size() - 1)) ? ptr[k] + 1 : ptr[k];
        else if (ret == 'k') // 'k' to move to previous cluster
        {
            k = (k > 0) ? k - 1 : K-1;
        }
        else if (ret == 'l' || ret == ' ')  // 'l' move to next cluster
            k = (k < (K - 1)) ? k + 1 : 0;
        else if (ret == 13) // 'ENTER' to assign the current cluster element as the representative
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
        else if (ret == 27) // 'ESC' to indicate the end of the manual annotation
            keep_selecting = false;
    }
    
    cv::destroyWindow("Viewer");

    // some irrelevant data re-organization

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
    
    // eventually, the computation of the intrinsics: camera_matrix and dist_coeffs

    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
    std::vector<cv::Mat> rvecs, tvecs; // will not be saved

    std::vector<std::vector<cv::Point2f> > image_points;
    uls::mat_to_vecvec<cv::Point2f>(corners_selection.reshape(2, corners_selection.rows), image_points);

    std::vector<std::vector<cv::Point3f> > object_points (1);
    uls::calcBoardCornerPositions(pattern_size, square_size.x, square_size.y, object_points[0]);
    object_points.resize(image_points.size(), object_points[0]);

    // ... calibrate!

    double rms = cv::calibrateCamera(object_points, image_points, frame_size, camera_matrix, dist_coeffs, rvecs, tvecs);
    std::cout << "RMS: " << rms << '\n'; // print the calibration error. 0.3 would be very good, 0.6 is acceptable, and above 1.0 it is not.

    // save the results into intrinsics_filepath

    cv::FileStorage fstorage_out (intrinsics_filepath, cv::FileStorage::WRITE);
    fstorage_out << "prefix" << prefix;
    fstorage_out << "corners" << corners_selection;
    fstorage_out << "corner_frames" << corner_frames_selection;
    fstorage_out << "camera_matrix" << camera_matrix;
    fstorage_out << "dist_coeffs" << dist_coeffs;
    fstorage_out << "rms" << rms;
    fstorage_out << "resize_dims" << frame_size;
    fstorage_out << "pattern_size" << pattern_size;
    fstorage_out << "square_size" << square_size;
    fstorage_out << "serial_number" << serial_number;
    fstorage_out.release();

    // visualize the results

    for (std::string frame_path : frames_all)
    {
        cv::Mat img = cv::imread(frame_path, cv::IMREAD_UNCHANGED);
        if (prefix == "pt/thermal")
            uls::thermal_to_8bit(img, img); 

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

