// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <iostream>
#include <librealsense2/rs.hpp>     // Include RealSense Cross Platform API
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>
#include <queue>
#include <boost/filesystem.hpp>
#include <ctime>   // localtime
#include <sstream> // stringstream
#include <iomanip> // put_time
#include <string>  // string
#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include "pt_pipeline.hpp"
#include "safe_queue.hpp"

bool debug = true;

namespace
{
    const size_t ERROR_UNHANDLED_EXCEPTION = -1;
    const size_t SUCCESS = 0;
    const size_t ERROR_IN_COMMAND_LINE = 1;
    const size_t ERROR_RS = 3;
    const size_t ERROR_PT = 4;
}

namespace po = boost::program_options;
namespace fs = boost::filesystem;

using timestamp_t = std::chrono::time_point<std::chrono::system_clock>;

//void remove_background(rs2::video_frame& other_frame, const rs2::depth_frame& depth_frame, float depth_scale, float clipping_dist)
//{
//    const uint16_t* p_depth_frame = reinterpret_cast<const uint16_t*>(depth_frame.get_data());
//    uint8_t* p_other_frame = reinterpret_cast<uint8_t*>(const_cast<void*>(other_frame.get_data()));
//
//    int width = other_frame.get_width();
//    int height = other_frame.get_height();
//    int other_bpp = other_frame.get_bytes_per_pixel();
//
//#pragma omp parallel for schedule(dynamic) //Using OpenMP to try to parallelise the loop
//    for (int y = 0; y < height; y++)
//    {
//        auto depth_pixel_index = y * width;
//        for (int x = 0; x < width; x++, ++depth_pixel_index)
//        {
//            // Get the depth value of the current pixel
//            auto pixels_distance = depth_scale * p_depth_frame[depth_pixel_index];
//
//            // Check if the depth value is invalid (<=0) or greater than the threashold
//            if (pixels_distance <= 0.f || pixels_distance > clipping_dist)
//            {
//                // Calculate the offset in other frame's buffer to current pixel
//                auto offset = depth_pixel_index * other_bpp;
//
//                // Set pixel to "background" color (0x999999)
//                std::memset(&p_other_frame[offset], 0x99, other_bpp);
//            }
//        }
//    }
//}

void upfront_cv_window_hack()
{
    cv::namedWindow("GetFocus", CV_WINDOW_NORMAL);
    cv::Mat img = cv::Mat::zeros(1, 1, CV_8UC3);
    cv::imshow("GetFocus", img);
    cv::setWindowProperty("GetFocus", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    //cv::waitKey(0);
    cv::setWindowProperty("GetFocus", CV_WND_PROP_FULLSCREEN, CV_WINDOW_NORMAL);
    cv::destroyWindow("GetFocus");
}

float get_depth_scale(rs2::device dev)
{
    // Go over the device's sensors
    for (rs2::sensor& sensor : dev.query_sensors())
    {
        // Check if the sensor if a depth sensor
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
        {
            return dpt.get_depth_scale();
        }
    }
    throw std::runtime_error("Device does not have a depth sensor");
}

rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams)
{
    //Given a vector of streams, we try to find a depth stream and another stream to align depth with.
    //We prioritize color streams to make the view look better.
    //If color is not available, we take another stream that (other than depth)
    rs2_stream align_to = RS2_STREAM_ANY;
    bool depth_stream_found = false;
    bool color_stream_found = false;
    for (rs2::stream_profile sp : streams)
    {
        rs2_stream profile_stream = sp.stream_type();
        if (profile_stream != RS2_STREAM_DEPTH)
        {
            if (!color_stream_found)         //Prefer color
                align_to = profile_stream;
            
            if (profile_stream == RS2_STREAM_COLOR)
            {
                color_stream_found = true;
            }
        }
        else
        {
            depth_stream_found = true;
        }
    }
    
    if(!depth_stream_found)
        throw std::runtime_error("No Depth stream available");
    
    if (align_to == RS2_STREAM_ANY)
        throw std::runtime_error("No stream found to align with Depth");
    
    return align_to;
}

bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev)
{
    for (auto&& sp : prev)
    {
        //If previous profile is in current (maybe just added another)
        auto itr = std::find_if(std::begin(current), std::end(current), [&sp](const rs2::stream_profile& current_sp) { return sp.unique_id() == current_sp.unique_id(); });
        if (itr == std::end(current)) //If it previous stream wasn't found in current
        {
            return true;
        }
    }
    return false;
}

/*
 * https://stackoverflow.com/questions/17223096/outputting-date-and-time-in-c-using-stdchrono
 */
std::string current_time_and_date()
{
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H.%M.%S");
    return ss.str();
}

cv::Mat thermal_to_8bit(cv::Mat data)
{
    cv::Mat img;
    double minVal, maxVal;
    cv::Point minIdx, maxIdx;
    
    cv::normalize(data, img, 0, 65535, cv::NORM_MINMAX);
    img.convertTo(img, CV_8U, 1/256.);
    
    return img;
}

cv::Mat depth_to_8bit(cv::Mat data)
{
    cv::Mat img;
    
    img.convertTo(img, CV_8U, 1/256.);
    
    return img;
}

void produce_realsense(rs2::pipeline & pipe, rs2::pipeline_profile & profile, SafeQueue<std::pair<std::pair<cv::Mat,cv::Mat>,timestamp_t>> & q, bool & is_capturing, double & elapsed, double duration)
{
    rs2_stream align_to = find_stream_to_align(profile.get_streams());
    rs2::align align(align_to);
    float depth_scale = get_depth_scale(profile.get_device());
    
//    rs2::colorizer color_map;
//    color_map.set_option(RS2_OPTION_HISTOGRAM_EQUALIZATION_ENABLED, 1.f);
//    color_map.set_option(RS2_OPTION_COLOR_SCHEME, 2.f); // White to Black
    
    while (is_capturing)
    {
        // Block program until frames arrive
        rs2::frameset frames = pipe.wait_for_frames();
        timestamp_t ts = std::chrono::system_clock::now();
        
        if (profile_changed(pipe.get_active_profile().get_streams(), profile.get_streams()))
        {
            //If the profile was changed, update the align object, and also get the new device's depth scale
            profile = pipe.get_active_profile();
            align_to = find_stream_to_align(profile.get_streams());
            align = rs2::align(align_to);
            depth_scale = get_depth_scale(profile.get_device());
        }
        
        //Get processed aligned frame
        auto processed = align.process(frames);
        
        // Trying to get both color and aligned depth frames
        rs2::video_frame other_frame = processed.first_or_default(align_to);
        rs2::depth_frame aligned_depth_frame = processed.get_depth_frame();

        // remove_background(other_frame, aligned_depth_frame, depth_scale, 2.0);  // debug
        
        // Print the distance
        cv::Mat c_ = cv::Mat(cv::Size(1280,720), CV_8UC3, const_cast<void *>(other_frame.get_data()), cv::Mat::AUTO_STEP);
        cv::Mat d_ = cv::Mat(cv::Size(1280,720), CV_16U, const_cast<void *>(aligned_depth_frame.get_data()), cv::Mat::AUTO_STEP);
        /* debug */
        //if (debug)
        //{
        //rs2::frame aligned_depth_col = color_map.process(aligned_depth_frame);
        //cv::Mat img_dcol_(cv::Size(1280, 720), CV_8UC3, (void*)aligned_depth_col.get_data(), cv::Mat::AUTO_STEP);
        //cv::imshow("depth dbg", img_c_);
        //cv::waitKey(1);
        //}
        /* end debug */

        q.enqueue(std::pair<std::pair<cv::Mat,cv::Mat>,timestamp_t>(std::pair<cv::Mat,cv::Mat>(c_.clone(),d_.clone()),ts));
        std::cerr << "[RS] <PRODUCER> enqueued " << q.size() << " ..." << '\n';
        
        if (debug)
        {
            cv::rectangle(c_, cv::Point(0,0), cv::Point(int(c_.cols*(elapsed/duration)),6), cv::Scalar(0,255,0), -1);
            cv::imshow("purthermal producer view", c_);
            cv::waitKey(1);
        }
    }
}

void produce_purethermal(pt::pipeline & p, SafeQueue<std::pair<cv::Mat,timestamp_t>> & q, bool & is_capturing, double & elapsed, double duration)
{
    while (is_capturing)
    {
        // Block program until frames arrive
        uvc_frame_t *frame = p.wait_for_frames();
        timestamp_t ts = std::chrono::system_clock::now();
        
        cv::Mat img_t_ = cv::Mat(cv::Size(160, 120), CV_16UC1, frame->data);
        cv::Mat img_t;
        cv::flip(img_t_, img_t, -1);
        q.enqueue(std::pair<cv::Mat,timestamp_t>(img_t,ts));
        std::cerr << "[PT] <PRODUCER> enqueued " << q.size() << " ..." << '\n';
        
        if (debug)
        {
            cv::Mat imgv_t = thermal_to_8bit(img_t);
            cv::cvtColor(imgv_t, imgv_t, cv::COLOR_GRAY2BGR);
            cv::resize(imgv_t, imgv_t, cv::Size(640,480));
            cv::rectangle(imgv_t, cv::Point(0,0), cv::Point(int(imgv_t.cols*(elapsed/duration)),6), cv::Scalar(0,255,0), -1);
            cv::imshow("purthermal producer view", imgv_t);
            cv::waitKey(1);
        }
    }
}

void consume_realsense(SafeQueue<std::pair<std::pair<cv::Mat,cv::Mat>,timestamp_t>> & q, bool & is_capturing, fs::path dir)
{
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);
    
    int fid = 0;
    boost::format fmt("%08d");
    std::ofstream outfile((dir / "rs.log").string(), std::ios_base::app);;
    while (is_capturing || q.size() > 0)
    {
        std::pair<std::pair<cv::Mat,cv::Mat>,timestamp_t> capture = q.dequeue();
        std::cout << "[RS] >CONSUMER< dequeued " << q.size() << " ..." << '\n';

        long time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(capture.second.time_since_epoch()).count();
        fmt % fid;
        outfile << fmt.str() << ',' << std::to_string(time_ms) << '\n';
        
        fs::path color_dir = dir / "rs/color/";
        fs::path depth_dir = dir / "rs/depth/";
        cv::imwrite((color_dir / ("c_" + fmt.str() + ".jpg")).string(), capture.first.first);
        cv::imwrite((depth_dir / ("d_" + fmt.str() + ".png")).string(), capture.first.second, compression_params);
        
        fid++;
    }
}

void consume_purethermal(SafeQueue<std::pair<cv::Mat,timestamp_t>> & q, bool & is_capturing, fs::path dir)
{
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);
    
    int fid = 0;
    boost::format fmt("%08d");
    std::ofstream outfile((dir / "pt.log").string(), std::ios_base::app);
    while (is_capturing || q.size() > 0)
    {
        std::pair<cv::Mat,timestamp_t> capture = q.dequeue();
        std::cout << "[PT] >CONSUMER< dequeued " << q.size() << " ..." << '\n';

        long time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(capture.second.time_since_epoch()).count();
        fmt % fid;
        outfile << fmt.str() << ',' << std::to_string(time_ms) << '\n';
        
        fs::path thermal_dir = dir / "pt/thermal/";
        cv::imwrite((thermal_dir / ("t_" + fmt.str() + ".png")).string(), capture.first, compression_params);

        fid++;
    }
}

void timer(bool & is_capturing, double & elapsed, double duration)
{
    elapsed = 0;
    auto start = std::chrono::high_resolution_clock::now();
    while (is_capturing)
    {
        // Control total time
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> chrono_elapsed = end-start;
        elapsed = chrono_elapsed.count();
        is_capturing = elapsed <= duration;
    }
}

int main(int argc, char * argv[]) try
{
    /* -------------------------------- */
    /*   Command line argument parsing  */
    /* -------------------------------- */
    
    po::options_description desc("Program options");
    desc.add_options()
        ("help,h", "Print help messages")
        ("duration,d", po::value<double>()->default_value(8000), "Duration of the recording in milliseconds (ms)")
        ("fps,s", po::value<int>()->default_value(30), "Acquisition speed (fps) of realsense (integer number 1~30)")
        ("output_dir,o", po::value<std::string>()->default_value("."), "Output directory to save acquired data");
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm); // can throw

    // --help option?
    if (vm.count("help"))
    {
        std::cout << "Basic Command Line Parameter App" << std::endl
        << desc << std::endl;
        return SUCCESS;
    }
    
    po::notify(vm); // throws on error, so do after help in case
    
    /* --------------- */
    /*   Actual code  */
    /* --------------- */
        
    // rs2::log_to_console(RS2_LOG_SEVERITY_DEBUG);

    // Create destiny directories to save acquired data
    std::string date_and_time = current_time_and_date();
    namespace fs = boost::filesystem;
    fs::path parent = fs::path(vm["output_dir"].as<std::string>()) / fs::path(date_and_time);
        
    boost::filesystem::create_directories(parent / fs::path("rs/color/"));
    boost::filesystem::create_directories(parent / fs::path("rs/depth/"));
//    boost::filesystem::create_directories(parent / fs::path("pt/thermal/"));
        
    // Initialize adquisition cues for consumer-producer setup
    double duration = vm["duration"].as<double>(); // acquisition time in ms
    SafeQueue<std::pair<std::pair<cv::Mat,cv::Mat>,timestamp_t>> queue_rs;
//    SafeQueue<std::pair<cv::Mat,timestamp_t>> queue_pt;
    bool is_capturing = false; // will be set to true later

    // Create a Pipeline - this serves as a top-level API for streaming and processing frames
    rs2::pipeline pipe_rs;
//    pt::pipeline pipe_pt;
    
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, vm["fps"].as<int>());
    cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, vm["fps"].as<int>());
    // cfg.enable_stream(RS2_STREAM_INFRARED, 1280, 720, RS2_FORMAT_Y8, 15);
    
    // Configure and start the pipeline
    rs2::pipeline_profile profile = pipe_rs.start(cfg);
//    pipe_pt.start(); // no need for external configuration
    
    // hacks
    upfront_cv_window_hack();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    is_capturing = true;
    std::thread c_rs_thr(consume_realsense, std::ref(queue_rs), std::ref(is_capturing), parent);
//    std::thread c_pt_thr(consume_purethermal, std::ref(queue_pt), std::ref(is_capturing), parent);
    
    /* Timer modifies the is_capturing flag after X millisecons */
    double elapsed_time = 0;
    std::thread timer_thr(timer, std::ref(is_capturing), std::ref(elapsed_time), duration);
    timer_thr.detach();
    
//    std::thread p_rs_thr(produce_realsense, std::ref(pipe_rs), std::ref(profile), std::ref(queue_rs), std::ref(is_capturing));
    produce_realsense(pipe_rs, profile, queue_rs, is_capturing, elapsed_time, duration);
//    produce_purethermal(pipe_pt, queue_pt, is_capturing, elapsed_time, duration); // if debug, imshow inside needs to run in main thread

//    p_rs_thr.join();
    // p_pt_thr.join();

    pipe_rs.stop();
//    pipe_pt.stop();

    /* Wait for consumer threads to finish */
    c_rs_thr.join();
//    c_pt_thr.join();

//    std::thread p_pt_thr(produce_purethermal, std::ref(pipe_pt), std::ref(queue_pt), std::ref(is_capturing), std::ref(elapsed_time), duration); // if debug, imshow inside needs to run in main thread
//    produce_realsense(pipe_rs, profile, queue_rs, is_capturing);
//
//    if (debug)
//        cv::destroyAllWindows();
////        cv::destroyWindow("purthermal producer view");
//
//    p_pt_thr.join();
////    p_rs_thr.join();
//
//    pipe_rs.stop();
//    pipe_pt.stop();
//
//    /* Wait for consumer threads to finish */
//    c_rs_thr.join();
//    c_pt_thr.join();
    
    std::cout << "Sequence saved in " << date_and_time << '\n';

    return SUCCESS;
}
catch(po::error& e)
{
    std::cerr << "Error parsing arguments: " << e.what() << std::endl;
    return ERROR_IN_COMMAND_LINE;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n " << e.what() << std::endl;
    return ERROR_RS;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return ERROR_UNHANDLED_EXCEPTION;
}

