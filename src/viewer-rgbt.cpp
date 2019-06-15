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
#include <boost/program_options.hpp>

#include "utils/common.hpp"
#include "utils/calibration.hpp"
#include "utils/synchronization.hpp"

bool debug = true;

namespace po = boost::program_options;
namespace fs = boost::filesystem;

namespace 
{ 
  const size_t ERROR_IN_COMMAND_LINE = 1; 
  const size_t SUCCESS = 0; 
  const size_t ERROR_UNHANDLED_EXCEPTION = 2; 
 
} // namespace

// std::vector<std::string> tokenize(std::string s, std::string delimiter)
// {
//     std::vector<std::string> tokens;

//     size_t pos = 0;
//     std::string token;
//     while ((pos = s.find(delimiter)) != std::string::npos) 
//     {
//         token = s.substr(0, pos);
//         tokens.push_back(token);
//         s.erase(0, pos + delimiter.length());
//     }
//     tokens.push_back(s); // last token

//     return tokens;
// }

// uls::Timestamp process_log_line(std::string line)
// {
//     std::vector<std::string> tokens = tokenize(line, ",");

//     uls::Timestamp ts;
//     ts.id = tokens.at(0);
//     std::istringstream iss (tokens.at(1));
//     iss >> ts.time;
    
//     return ts;
// }
// 
// std::vector<uls::Timestamp> read_log(fs::path log_path)
// {
//     std::ifstream log (log_path.string());
//     std::string line;
//     std::vector<uls::Timestamp> tokenized_lines;
//     if (log.is_open()) {
//         while (std::getline(log, line)) {
//             uls::Timestamp ts = process_log_line(line);
//             tokenized_lines.push_back(ts);
//         }
//         log.close();
//     }

//     return tokenized_lines;
// }

int main(int argc, char * argv[]) try
{
    /* -------------------------------- */
    /*   Command line argument parsing  */
    /* --------------   ------------------ */
    
    std::string input_path_str;
    bool loop = false;

    po::options_description desc("Program options");
    desc.add_options()
        ("help,h", "Print help messages")
        ("fps,f", po::value<float>()->default_value(0.f), "Acquisition speed (fps) of realsense (integer number 1~30)")
        ("loop,", po::bool_switch(&loop), "Loop the sequence visualization")
        ("sync-delay,s", po::value<int>()->default_value(30), "Number of milliseconds (ms) del")
        ("calibration-file", po::value<std::string>()->default_value(""), "Calibration mapping parameters")
        ("input-dir", po::value<std::string>(&input_path_str)->required(), "Input directory containing rs/pt frames and timestamp files");
    
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
        
    fs::path input_path (vm["input-dir"].as<std::string>());

    fs::path info_path (input_path / "rs_info.yml");

    fs::path color_path (input_path / "rs/color/");
    fs::path depth_path (input_path / "rs/depth/");
    fs::path therm_path (input_path / "pt/thermal/");

    fs::path log_color_path (input_path / "rs/color.log");
    fs::path log_depth_path (input_path / "rs/depth.log");
    fs::path log_thermal_path (input_path / "pt/thermal.log");

    fs::path calib_filepath;
    fs::path calib_file (vm["calibration-file"].as<std::string>());
    if (!calib_file.empty())
        calib_filepath = input_path / "calibration.yml";

    // cv::namedWindow("Viewer");
    float fps = vm["fps"].as<float>();
    int wait_time = 0;
    if (fps > 0.f) 
        wait_time = (1./fps) * 1000.;

    cv::FileStorage fs;

    std::string rs_serial_number;
    float depth_scale;
    if (fs.open(info_path.string(), cv::FileStorage::READ))
    {
        fs["rs-serial_number"] >> rs_serial_number;
        fs["rs-depth_scale"] >> depth_scale;
        fs.release();
    }

    std::shared_ptr<std::map<std::string,uls::intrinsics_t> > intrinsics;
    std::shared_ptr<uls::extrinsics_t> extrinsics;
    if (fs.open(calib_filepath.string(), cv::FileStorage::READ))
    {
        std::string modality_1, modality_2;
        fs["modality-1"] >> modality_1;
        fs["modality-2"] >> modality_2;
        
        intrinsics = std::make_shared<std::map<std::string,uls::intrinsics_t> >();
        fs["camera_matrix-1"] >> (*intrinsics)[modality_1].camera_matrix;
        fs["camera_matrix-2"] >> (*intrinsics)[modality_2].camera_matrix;
        fs["dist_coeffs-1"]   >> (*intrinsics)[modality_1].dist_coeffs;
        fs["dist_coeffs-2"]   >> (*intrinsics)[modality_2].dist_coeffs;

        extrinsics = std::make_shared<uls::extrinsics_t>();
        fs["R"] >> extrinsics->R;
        fs["T"] >> extrinsics->T;
        fs.release();
    }

    // read logs
    std::vector<uls::Timestamp> log_color = uls::read_log(log_color_path);
    std::vector<uls::Timestamp> log_depth = uls::read_log(log_depth_path);
    std::vector<uls::Timestamp> log_thermal = uls::read_log(log_thermal_path);

    // std::vector<std::pair<uls::Timestamp,uls::Timestamp> > log_synced;
    // uls::time_sync(log_color, log_thermal, log_synced, 50);
    std::vector<int> inds_color, inds_depth, inds_thermal;
    uls::time_sync(log_color, log_thermal, inds_color, inds_thermal, 50);
    uls::time_sync(log_depth, log_thermal, inds_depth, inds_thermal, 50);
    assert(inds_color.size() == inds_depth.size());

    cv::namedWindow("Viewer");
    do
    {
        for (int i = 0; i < inds_color.size() - 1; i++)
        // for (int i = 0; i < log_synced.size() - 1; i++)
        {
            // get synced frame timestamp in the multiple modalities
            // uls::Timestamp rs_ts = log_synced[i].first;
            // uls::Timestamp pt_ts = log_synced[i].second;
            uls::Timestamp color_ts = log_color[inds_color[i]];
            uls::Timestamp depth_ts = log_depth[inds_depth[i]];
            uls::Timestamp thermal_ts = log_thermal[inds_thermal[i]];

            // load color image
            cv::Mat c = cv::imread((color_path / color_ts.id).string(), cv::IMREAD_UNCHANGED);
            
            // load depth image and prepare 8-bit colormapped version for visualization
            cv::Mat d, d8;
            d = cv::imread((depth_path / depth_ts.id).string(), cv::IMREAD_UNCHANGED);
            d.setTo(0, d > 5000); // and cut at X mm of depth
            uls::depth_to_8bit(d, d8, cv::COLORMAP_BONE);
            
            // load thermal image and prepare 8-bit colormapped version for visualization
            cv::Mat t, t8;
            t = cv::imread((therm_path / thermal_ts.id).string(), cv::IMREAD_UNCHANGED);
            cv::Rect t_roi = uls::resize(t, t, d.size()); // resize to same dimensions and pad margins to match aspect ratio
            uls::thermal_to_8bit(t, t8, t_roi);

            cv::Mat ta; // thermal aligned
            
            // calibration
            if (intrinsics != nullptr && extrinsics != nullptr)
            {
                // initialize 8-bit images
                cv::Mat cu  ( c.size(),  c.type());
                cv::Mat d8u (d8.size(), d8.type());
                cv::Mat t8u (t8.size(), t8.type());
                // undistort using intrinsics         
                cv::undistort(c, cu, (*intrinsics)["rs/color"].camera_matrix, (*intrinsics)["rs/color"].dist_coeffs);
                cv::undistort(d8, d8u, (*intrinsics)["rs/color"].camera_matrix, (*intrinsics)["rs/color"].dist_coeffs);
                cv::undistort(t8, t8u, (*intrinsics)["pt/thermal"].camera_matrix, (*intrinsics)["pt/thermal"].dist_coeffs);
                // 1/2. compute mapping to perfom alignment thermal -> color/depth
                cv::Mat map_x, map_y;
                cv::Mat du (d.size(), d.type());
                cv::undistort(d, du, (*intrinsics)["rs/color"].camera_matrix, (*intrinsics)["rs/color"].dist_coeffs);
                // 2/2. and use extrinsics for that
                align_to_depth(du, (*intrinsics)["rs/color"].camera_matrix, (*intrinsics)["pt/thermal"].camera_matrix, depth_scale, extrinsics, map_x, map_y);

                // finally, map undistorted thermal to color/depth
                cv::Mat t8a;
                t8a.create(t8u.size(), t8u.type());
                cv::remap(t8u, t8a, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

                // since calibration has been performed, substitute non-calibrated images (for visualization)
                c  = cu;
                d  = d8u;
                t  = t8u;
                ta = t8a;
            }
            
            // tile frame images in a mosaic
            std::vector<cv::Mat> frames = {c, d, t, ta};
            cv::Mat tiling;
            uls::tile(frames, 1280, 720, 2, 2, tiling);

            // visualize mosaic
            cv::imshow("Viewer", tiling);
            
            cv::waitKey(wait_time);
            // cv::waitKey(wait_time > 0 ? wait_time : log_synced[i+1].first.time - rs_ts.time);
        }
    } while (loop);
    
    cv::destroyWindow("Viewer");

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

