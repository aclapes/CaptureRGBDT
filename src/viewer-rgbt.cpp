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

std::vector<std::string> tokenize(std::string s, std::string delimiter)
{
    std::vector<std::string> tokens;

    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) 
    {
        token = s.substr(0, pos);
        tokens.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    tokens.push_back(s); // last token

    return tokens;
}

uls::Timestamp process_log_line(std::string line)
{
    std::vector<std::string> tokens = tokenize(line, ",");

    uls::Timestamp ts;
    ts.id = tokens.at(0);
    std::istringstream iss (tokens.at(1));
    iss >> ts.time;
    
    return ts;
}

std::vector<uls::Timestamp> read_log_file(fs::path log_path)
{
    std::ifstream log (log_path.string());
    std::string line;
    std::vector<uls::Timestamp> tokenized_lines;
    if (log.is_open()) {
        while (std::getline(log, line)) {
            uls::Timestamp ts = process_log_line(line);
            tokenized_lines.push_back(ts);
        }
        log.close();
    }

    return tokenized_lines;
}

int main(int argc, char * argv[]) try
{
    /* -------------------------------- */
    /*   Command line argument parsing  */
    /* --------------   ------------------ */
    
    std::string input_dir_str;

    po::options_description desc("Program options");
    desc.add_options()
        ("help,h", "Print help messages")
        ("fps,f", po::value<float>()->default_value(0.f), "Acquisition speed (fps) of realsense (integer number 1~30)")
        ("calibration-params", po::value<std::string>()->default_value(""), "Calibration mapping parameters")
        ("input-dir", po::value<std::string>(&input_dir_str)->required(), "Input directory containing rs/pt frames and timestamp files");
    
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

    fs::path info_path (input_dir / fs::path("rs_info.yml"));

    fs::path log_rs_path (input_dir / fs::path("rs.log"));
    fs::path log_pt_path (input_dir / fs::path("pt.log"));

    fs::path color_path (input_dir / fs::path("rs/color/"));
    fs::path depth_path (input_dir / fs::path("rs/depth/"));
    fs::path therm_path (input_dir / fs::path("pt/thermal/"));

    fs::path calib_path (vm["calibration-params"].as<std::string>());
    if (calib_path.empty())
        calib_path = input_dir / fs::path("calibration.yml");

    // cv::namedWindow("Viewer");
    float fps = vm["fps"].as<float>();
    int wait_time = 0;
    if (fps > 0.f) 
        wait_time = (1./fps) * 1000.;

    std::vector<uls::Timestamp> log_rs = read_log_file(log_rs_path);
    std::vector<uls::Timestamp> log_pt = read_log_file(log_pt_path);

    std::vector<std::pair<uls::Timestamp,uls::Timestamp> > log_synced;
    uls::time_sync(log_rs, log_pt, log_synced, 250);

    cv::FileStorage fs;

    std::string rs_serial_number;
    float depth_scale;
    if (fs.open(info_path.string(), cv::FileStorage::READ))
    {
        fs["serial_number"] >> rs_serial_number;
        fs["depth_scale"] >> depth_scale;
        fs.release();
    }

    std::shared_ptr<std::map<std::string,uls::intrinsics_t> > intrinsics;
    std::shared_ptr<uls::extrinsics_t> extrinsics;
    if (fs.open(calib_path.string(), cv::FileStorage::READ))
    {
        std::string modality_1, modality_2;
        fs["modality-1"] >> modality_1;
        fs["modality-2"] >> modality_2;
        
        intrinsics = std::make_shared<std::map<std::string,uls::intrinsics_t> >();
        fs["camera_matrix_1"] >> (*intrinsics)[modality_1].camera_matrix;
        fs["camera_matrix_2"] >> (*intrinsics)[modality_2].camera_matrix;
        fs["dist_coeffs_1"]   >> (*intrinsics)[modality_1].dist_coeffs;
        fs["dist_coeffs_2"]   >> (*intrinsics)[modality_2].dist_coeffs;

        extrinsics = std::make_shared<uls::extrinsics_t>();
        fs["R"] >> extrinsics->R;
        fs["T"] >> extrinsics->T;
        fs.release();
    }


    for (int i = 0; i < log_synced.size() - 1; i++)
    {
        // read timestamps
        uls::Timestamp rs_ts = log_synced[i].first;
        uls::Timestamp pt_ts = log_synced[i].second;

        // read frames
        uls::ColorFrame   cf (color_path / fs::path("c_" + rs_ts.id + ".jpg"));
        uls::DepthFrame   df (depth_path / fs::path("d_" + rs_ts.id + ".png"));
        uls::ThermalFrame tf (therm_path / fs::path("t_" + pt_ts.id + ".png"), cv::Size(1280,720));

        // preprocess frames
        // d = uls::DepthFrame::to_8bit(d, cv::COLORMAP_JET);
        // cv::cvtColor(tf.mat(), t, cv::COLOR_GRAY2BGR);

        cv::Mat c,t,d,ta,cdt;
        if (intrinsics != nullptr && extrinsics != nullptr)
        {
            // c = cf.mat().clone();
            c.create(cf.mat().size(), cf.mat().type());
            cv::undistort(cf.mat(), c, (*intrinsics)["Color"].camera_matrix, (*intrinsics)["Color"].dist_coeffs);
            // d = df.mat().clone();
            d.create(df.mat().size(), df.mat().type());
            cv::undistort(df.mat(), d, (*intrinsics)["Color"].camera_matrix, (*intrinsics)["Color"].dist_coeffs);
            // t = tf.mat().clone();
            t.create(tf.mat().size(), tf.mat().type());
            cv::undistort(tf.mat(), t, (*intrinsics)["Thermal"].camera_matrix, (*intrinsics)["Thermal"].dist_coeffs);

            cv::Mat map_x, map_y;
            align_to_depth(d, (*intrinsics)["Color"].camera_matrix, (*intrinsics)["Thermal"].camera_matrix, depth_scale, extrinsics, map_x, map_y);
            // std::cout << map_x << std::endl;
            ta.create(t.size(), t.type());
            cv::remap(t, ta, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
        }
        
        // tile frame images in a mosaic
        std::vector<cv::Mat> frames = {c, uls::DepthFrame::to_8bit(d, cv::COLORMAP_JET), t, ta};
        cv::Mat tiling;
        uls::tile(frames, 1280, 720, 2, 2, tiling);

        // visualize mosaic
        cv::imshow("Viewer", tiling);
        cv::waitKey(wait_time > 0 ? wait_time : log_synced[i+1].first.time - rs_ts.time);
    }
    // cv::destroyWindow("Viewer");

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

