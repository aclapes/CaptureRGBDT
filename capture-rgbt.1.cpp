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
#include <boost/algorithm/string.hpp>

#include "pt_pipeline.hpp"
#include "safe_queue.hpp"
#include "utils.hpp"

bool debug = true;

#define USE_COLOR   static_cast<int>( 1 << 0 )
#define USE_DEPTH   static_cast<int>( 1 << 1 )
#define USE_THERM   static_cast<int>( 1 << 2 )

#define FIND_PATTERN_ON_COLOR   static_cast<int>( 1 << 0 )
#define FIND_PATTERN_ON_THERM   static_cast<int>( 1 << 1 )

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

// using timestamp_t = std::chrono::time_point<std::chrono::system_clock>;

typedef struct
{
    cv::Mat img_c; // color
    cv::Mat img_d; // depth
    cv::Mat img_ir; // ir
    std::chrono::time_point<std::chrono::system_clock> ts;
} rs_frame_t;

typedef struct
{
    cv::Mat img; // thermal
    std::chrono::time_point<std::chrono::system_clock> ts;
} pt_frame_t;

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

// cv::Mat thermal_to_8bit(cv::Mat data)
// {
//     cv::Mat img;
//     double minVal, maxVal;
//     cv::Point minIdx, maxIdx;
    
//     cv::normalize(data, img, 0, 65535, cv::NORM_MINMAX);
//     img.convertTo(img, CV_8U, 1/256.);
    
//     return img;
// }

// cv::Mat depth_to_8bit(cv::Mat data)
// {
//     cv::Mat img;
    
//     img.convertTo(img, CV_8U, 1/256.);
    
//     return img;
// }

// void produce_realsense(rs2::pipeline & pipe, rs2::pipeline_profile & profile, SafeQueue<std::pair<std::pair<cv::Mat,cv::Mat>,timestamp_t>> & q, bool & is_capturing, bool verbose)
// {
//     rs2_stream align_to = find_stream_to_align(profile.get_streams());
//     rs2::align align(align_to);
//     float depth_scale = get_depth_scale(profile.get_device());
    
//     /*
//     rs2::colorizer color_map;
//     color_map.set_option(RS2_OPTION_HISTOGRAM_EQUALIZATION_ENABLED, 1.f);
//     color_map.set_option(RS2_OPTION_COLOR_SCHEME, 2.f); // White to Black
//     */
//     while (is_capturing)
//     {
//         rs2::frameset frames = pipe.wait_for_frames(); // blocking instruction
//         timestamp_t ts = std::chrono::system_clock::now();
        
//         if (profile_changed(pipe.get_active_profile().get_streams(), profile.get_streams()))
//         {
//             // If the profile was changed, update the align object, and also get the new device's depth scale
//             profile = pipe.get_active_profile();
//             align_to = find_stream_to_align(profile.get_streams());
//             align = rs2::align(align_to);
//             depth_scale = get_depth_scale(profile.get_device());
//         }

//         // Get processed aligned frame
//         rs2::frameset processed = align.process(frames);

//         // Trying to get both color and aligned depth frames
//         rs2::video_frame other_frame = processed.first_or_default(align_to);
//         rs2::depth_frame aligned_depth_frame = processed.get_depth_frame();
        
//         // Print the distance
//         cv::Mat c = cv::Mat(cv::Size(1280,720), CV_8UC3, const_cast<void *>(other_frame.get_data()), cv::Mat::AUTO_STEP);
//         cv::Mat d = cv::Mat(cv::Size(1280,720), CV_16U, const_cast<void *>(aligned_depth_frame.get_data()), cv::Mat::AUTO_STEP);

//         q.enqueue(std::pair<std::pair<cv::Mat,cv::Mat>,timestamp_t>(std::pair<cv::Mat,cv::Mat>(c.clone(),d.clone()),ts));
//         if (verbose)
//             std::cerr << "[RS] <PRODUCER> enqueued " << q.size() << " ..." << '\n';
//     }
// }

void produce_realsense(rs2::pipeline & pipe, rs2::pipeline_profile & profile, SafeQueue<rs_frame_t> & q, bool & is_capturing, int modality_flags, bool verbose)
{
    rs2_stream align_to;
    float depth_scale;
    rs2::align* p_align = NULL;

    if (modality_flags & USE_DEPTH)
    {
        align_to = find_stream_to_align(profile.get_streams());
        p_align = new rs2::align(align_to);
        depth_scale = get_depth_scale(profile.get_device());
    }
    
    /*
    rs2::colorizer color_map;
    color_map.set_option(RS2_OPTION_HISTOGRAM_EQUALIZATION_ENABLED, 1.f);
    color_map.set_option(RS2_OPTION_COLOR_SCHEME, 2.f); // White to Black
    */
    while (is_capturing)
    {
        rs2::frameset frames = pipe.wait_for_frames(); // blocking instruction
        
        rs_frame_t frame;
        frame.ts = std::chrono::system_clock::now();
        
        if (((modality_flags & USE_COLOR) | (modality_flags & USE_DEPTH)) == (USE_COLOR | USE_DEPTH)) 
        {
           if (profile_changed(pipe.get_active_profile().get_streams(), profile.get_streams()))
            {
                // If the profile was changed, update the align object, and also get the new device's depth scale
                profile = pipe.get_active_profile();
                align_to = find_stream_to_align(profile.get_streams());
                if (p_align != NULL) 
                    delete p_align;
                p_align = new rs2::align(align_to);
                depth_scale = get_depth_scale(profile.get_device());
            }

            // Get processed aligned frame
            rs2::frameset processed = p_align->process(frames);

            // Trying to get both color and aligned depth frames
            rs2::video_frame other_frame = processed.first_or_default(align_to);
            rs2::depth_frame aligned_depth_frame = processed.get_depth_frame();
        
            // Print the distance
            frame.img_c = cv::Mat(cv::Size(other_frame.get_width(), other_frame.get_height()), 
                                  CV_8UC3, 
                                  const_cast<void *>(other_frame.get_data()), 
                                  cv::Mat::AUTO_STEP).clone();
            frame.img_d = cv::Mat(cv::Size(aligned_depth_frame.get_width(), aligned_depth_frame.get_height()), 
                                  CV_16U, 
                                  const_cast<void *>(aligned_depth_frame.get_data()), 
                                  cv::Mat::AUTO_STEP).clone();
        }
        else if (modality_flags & USE_COLOR)
        {
            // Trying to get both color and aligned depth frames
            rs2::video_frame color_frame = frames.get_color_frame();
            
            // Print the distance
            frame.img_c = cv::Mat(cv::Size(color_frame.get_width(), color_frame.get_height()), 
                                  CV_8UC3, 
                                  const_cast<void *>(color_frame.get_data()), 
                                  cv::Mat::AUTO_STEP).clone();
        }
        else if (modality_flags & USE_DEPTH)
        {
            // Trying to get both color and aligned depth frames
            rs2::depth_frame depth_frame = frames.get_depth_frame();
            
            // Print the distance
            frame.img_d = cv::Mat(cv::Size(depth_frame.get_width(), depth_frame.get_height()), 
                                  CV_16U, 
                                  const_cast<void *>(depth_frame.get_data()), 
                                  cv::Mat::AUTO_STEP).clone();
        }

        q.enqueue( frame );
        if (verbose)
            std::cerr << "[RS] <PRODUCER> enqueued " << q.size() << " ..." << '\n';
    }

    if (p_align != NULL) 
        delete p_align;
}

// void produce_realsense_depth(rs2::pipeline & pipe, rs2::pipeline_profile & profile, SafeQueue<std::pair<cv::Mat,timestamp_t>> & q, bool & is_capturing, bool verbose)
// {
//     while (is_capturing)
//     {
//         rs2::frameset frames = pipe.wait_for_frames(); // blocking instruction
//         timestamp_t ts = std::chrono::system_clock::now();
        
//         // Trying to get both color and aligned depth frames
//         rs2::depth_frame depth_frame = frames.get_depth_frame();
        
//         // Print the distance
//         cv::Mat d = cv::Mat(cv::Size(depth_frame.get_width(),depth_frame.get_height()), 
//                             CV_16U, 
//                             const_cast<void *>(depth_frame.get_data()), 
//                             cv::Mat::AUTO_STEP);

//         q.enqueue(std::pair<cv::Mat,timestamp_t>(d.clone(),ts));
//         if (verbose)
//             std::cerr << "[RS] <PRODUCER> enqueued " << q.size() << " ..." << '\n';
//     }
// }

// void produce_realsense_color(rs2::pipeline & pipe, rs2::pipeline_profile & profile, SafeQueue<std::pair<cv::Mat,timestamp_t>> & q, bool & is_capturing, bool verbose)
// {
//     while (is_capturing)
//     {
//         rs2::frameset frames = pipe.wait_for_frames(); // blocking instruction
//         timestamp_t ts = std::chrono::system_clock::now();
        
//         // Trying to get both color and aligned depth frames
//         rs2::video_frame color_frame = frames.get_color_frame();
        
//         // Print the distance
//         cv::Mat c = cv::Mat(cv::Size(color_frame.get_width(),color_frame.get_height()), 
//                             CV_8UC3, 
//                             const_cast<void *>(color_frame.get_data()), 
//                             cv::Mat::AUTO_STEP);

//         q.enqueue(std::pair<cv::Mat,timestamp_t>(c.clone(),ts));
//         if (verbose)
//             std::cerr << "[RS] <PRODUCER> enqueued " << q.size() << " ..." << '\n';
//     }
// }


// void produce_realsense(rs2::pipeline & pipe, rs2::pipeline_profile & profile, SafeQueue<std::pair<std::pair<cv::Mat,cv::Mat>,timestamp_t>> & q, bool & is_capturing, bool verbose)
// {
//     rs2_stream align_to = find_stream_to_align(profile.get_streams());
//     rs2::align align(align_to);
//     float depth_scale = get_depth_scale(profile.get_device());

//     while (is_capturing)
//     {
//         rs2::frameset frames = pipe.wait_for_frames(); // blocking instruction
//         timestamp_t ts = std::chrono::system_clock::now();
        
//         if (profile_changed(pipe.get_active_profile().get_streams(), profile.get_streams()))
//         {
//             // If the profile was changed, update the align object, and also get the new device's depth scale
//             profile = pipe.get_active_profile();
//             align_to = find_stream_to_align(profile.get_streams());
//             align = rs2::align(align_to);
//             depth_scale = get_depth_scale(profile.get_device());
//         }

//         // Get processed aligned frame
//         rs2::frameset processed = align.process(frames);

//         // Trying to get both color and aligned depth frames
//         rs2::video_frame other_frame = processed.first_or_default(align_to);
//         rs2::depth_frame aligned_depth_frame = processed.get_depth_frame();
        
//         // Print the distance
//         cv::Mat c = cv::Mat(cv::Size(1280,720), CV_8UC3, const_cast<void *>(other_frame.get_data()), cv::Mat::AUTO_STEP);
//         cv::Mat d = cv::Mat(cv::Size(1280,720), CV_16U, const_cast<void *>(aligned_depth_frame.get_data()), cv::Mat::AUTO_STEP);

//         q.enqueue(std::pair<std::pair<cv::Mat,cv::Mat>,timestamp_t>(std::pair<cv::Mat,cv::Mat>(c.clone(),d.clone()),ts));
//         if (verbose)
//             std::cerr << "[RS] <PRODUCER> enqueued " << q.size() << " ..." << '\n';
//     }
// }

void produce_purethermal(pt::pipeline & p, SafeQueue<pt_frame_t> & q, bool & is_capturing, bool verbose)
{
    while (is_capturing)
    {
        // Block program until frames arrive
        uvc_frame_t *img = p.wait_for_frames();

        pt_frame_t frame;
        frame.ts = std::chrono::system_clock::now();
        frame.img = cv::Mat(cv::Size(160, 120), CV_16UC1, img->data).clone();
        // cv::flip(img_t, img_t, -1);
        q.enqueue( frame );

        if (verbose)
            std::cerr << "[PT] <PRODUCER> enqueued " << q.size() << " ..." << '\n';
    }
}

void consume_realsense(SafeQueue<rs_frame_t> & q, bool & is_capturing, fs::path dir, bool verbose)
{
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);
    
    int fid = 0;
    boost::format fmt("%08d");
    std::ofstream outfile;

    if (!dir.empty())
        outfile.open((dir / "rs.log").string(), std::ios_base::app);

    while (is_capturing || q.size() > 0)
    {
        rs_frame_t capture = q.dequeue();
        if (verbose)
            std::cout << "[RS] >CONSUMER< dequeued " << q.size() << " ..." << '\n';

        if (!dir.empty())
        {
            long time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(capture.ts.time_since_epoch()).count();
            fmt % fid;
            outfile << fmt.str() << ',' << std::to_string(time_ms) << '\n';
        
            fs::path color_dir = dir / "rs/color/";
            fs::path depth_dir = dir / "rs/depth/";
            if (!capture.img_c.empty())
                cv::imwrite((color_dir / ("c_" + fmt.str() + ".jpg")).string(), capture.img_c);
            if (!capture.img_d.empty())
                cv::imwrite((depth_dir / ("d_" + fmt.str() + ".png")).string(), capture.img_d, compression_params);
        }
        fid++;
    }
}

// void consume_realsense_color(SafeQueue<std::pair<cv::Mat,timestamp_t>> & q, bool & is_capturing, fs::path dir, bool verbose)
// {
//     std::vector<int> compression_params;
//     compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
//     compression_params.push_back(0);
    
//     int fid = 0;
//     boost::format fmt("%08d");
//     std::ofstream outfile;

//     if (!dir.empty())
//         outfile.open((dir / "rs.log").string(), std::ios_base::app);

//     while (is_capturing || q.size() > 0)
//     {
//         std::pair<cv::Mat,timestamp_t> capture = q.dequeue();
//         if (verbose)
//             std::cout << "[RS] >CONSUMER< dequeued " << q.size() << " ..." << '\n';

//         if (!dir.empty())
//         {
//             long time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(capture.second.time_since_epoch()).count();
//             fmt % fid;
//             outfile << fmt.str() << ',' << std::to_string(time_ms) << '\n';
        
//             fs::path color_dir = dir / "rs/color/";
//             cv::imwrite((color_dir / ("c_" + fmt.str() + ".jpg")).string(), capture.first);
//         }
//         fid++;
//     }
// }

// void consume_realsense_depth(SafeQueue<std::pair<cv::Mat,timestamp_t>> & q, bool & is_capturing, fs::path dir, bool verbose)
// {
//     std::vector<int> compression_params;
//     compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
//     compression_params.push_back(0);
    
//     int fid = 0;
//     boost::format fmt("%08d");
//     std::ofstream outfile;

//     if (!dir.empty())
//         outfile.open((dir / "rs.log").string(), std::ios_base::app);

//     while (is_capturing || q.size() > 0)
//     {
//         std::pair<cv::Mat,timestamp_t> capture = q.dequeue();
//         if (verbose)
//             std::cout << "[RS] >CONSUMER< dequeued " << q.size() << " ..." << '\n';

//         if (!dir.empty())
//         {
//             long time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(capture.second.time_since_epoch()).count();
//             fmt % fid;
//             outfile << fmt.str() << ',' << std::to_string(time_ms) << '\n';
        
//             fs::path color_dir = dir / "rs/depth/";
//             cv::imwrite((color_dir / ("d_" + fmt.str() + ".png")).string(), capture.first);
//         }
//         fid++;
//     }
// }

void consume_purethermal(SafeQueue<pt_frame_t> & q, bool & is_capturing, fs::path dir, bool verbose)
{
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);
    
    int fid = 0;
    boost::format fmt("%08d");
    std::ofstream outfile;
    
    if (!dir.empty())
        outfile.open((dir / "pt.log").string(), std::ios_base::app);
    
    while (is_capturing || q.size() > 0)
    {
        pt_frame_t capture = q.dequeue();
        if (verbose)
            std::cout << "[PT] >CONSUMER< dequeued " << q.size() << " ..." << '\n';

        if (!dir.empty())
        {
            long time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(capture.ts.time_since_epoch()).count();
            fmt % fid;
            outfile << fmt.str() << ',' << std::to_string(time_ms) << '\n';
            
            fs::path thermal_dir = dir / "pt/thermal/";
            cv::imwrite((thermal_dir / ("t_" + fmt.str() + ".png")).string(), capture.img, compression_params);
        }

        fid++;
    }
}

void timer(int duration, bool & is_capturing)
{
    auto start = std::chrono::steady_clock::now();

    is_capturing = true;
    while (is_capturing)
    {
        if (std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - start) >= std::chrono::milliseconds(duration))
            is_capturing = false;
    }
}

// void rs2_hardware_reset(int wait_time_ms)
// {
//     // Create a Pipeline - this serves as a top-level API for streaming and processing frames
//     rs2::context ctx;
//     auto list = ctx.query_devices(); // Get a snapshot of currently connected devices
//     if (list.size() == 0) 
//         throw std::runtime_error("No device detected. Is it plugged in?");
//     rs2::device dev = list.front();
//     dev.hardware_reset();
//     std::this_thread::sleep_for(std::chrono::milliseconds(wait_time_ms));
// }

void countdown(int total, int warn_every, bool verbose = true)
{
    while (total > 0) 
    {
        if (verbose) std::cout << total << '\n';
        std::this_thread::sleep_for(std::chrono::seconds(warn_every));
        total -= warn_every;
    }
}

void find_and_draw_chessboard(cv::Mat & img, cv::Size pattern_size, int flags = 0)
{
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::Mat corners;
    bool found = cv::findChessboardCorners(gray, pattern_size, corners, flags);

    cv::drawChessboardCorners(img, pattern_size, corners, found);
}


void visualize(std::string win_name,
               SafeQueue<rs_frame_t> & queue_rs, 
               SafeQueue<pt_frame_t> & queue_pt, 
               bool & is_capturing, 
               cv::Size pattern_size, 
               int find_pattern_flags,
               bool find_patterns = false)
{
    // Open visualization window
    cv::namedWindow(win_name);

    // If peek fails for some stream, the previous frame is shown.
    std::vector<cv::Mat> frames (3); // color, depth, thermal
    while (is_capturing)
    {
        try 
        {
            rs_frame_t rs_frame = queue_rs.peek();
            if (!rs_frame.img_c.empty())
            {
                frames[0] = rs_frame.img_c;
                // if (find_pattern_flags & FIND_PATTERN_ON_COLOR) 
                // {
                //     cv::resize(frames[0],frames[0],cv::Size(640,360));
                //     find_and_draw_chessboard(frames[0], pattern_size, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
                // }
            }
            if (!rs_frame.img_d.empty())
            {
                cv::Mat tmp = uls::DepthFrame::cut_at(rs_frame.img_d, 4000);
                frames[1] = uls::DepthFrame::to_8bit(tmp, cv::COLORMAP_JET);
            }
        }
        catch (const std::runtime_error & e)
        {
            std::cout << e.what() << '\n';
        }
        
        try
        {
            pt_frame_t pt_frame = queue_pt.peek();
            if (!pt_frame.img.empty())
            {
                frames[2] = uls::ThermalFrame::to_8bit(pt_frame.img);
                cv::cvtColor(frames[2], frames[2], cv::COLOR_GRAY2BGR);
                // if (find_pattern_flags & FIND_PATTERN_ON_THERM)
                //     find_and_draw_chessboard(frames[2], pattern_size, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
            }
        }
        catch (const std::runtime_error & e)
        {
            std::cerr << e.what() << '\n';
        }

        cv::Mat tiling;
        uls::tile(frames, 426, 240*frames.size(), 1, frames.size(), tiling);

        cv::imshow(win_name, tiling);
        cv::waitKey(1);
    }

    cv::destroyWindow(win_name);
}

// void visualize(std::string win_name,
//                SafeQueue<rs_frame_t> & queue_rs, 
//                SafeQueue<pt_frame_t> & queue_pt, 
//                bool & is_capturing, 
//                int modality_flags, 
//                cv::Size pattern_size, 
//                bool find_patterns = false)
// {
//     // Open visualization window
//     cv::namedWindow(win_name);

//     // If peek fails for some stream, the previous frame is shown.
//     std::vector<cv::Mat> frames (3); // color, depth, thermal
//     while (is_capturing)
//     {
//         if (modality_flags & USE_COLOR & USE_DEPTH)
//         {
//             try
//             {
//                 std::pair<std::pair<cv::Mat,cv::Mat>,timestamp_t> rs_peek = queue_rs.peek();
//                 frames[0] = rs_peek.first.first;
//                 frames[1] = uls::DepthFrame::to_8bit(rs_peek.first.second, cv::COLORMAP_JET);
//                 if (find_patterns) {
//                     cv::resize(frames[0],frames[0],cv::Size(640,360));
//                     find_and_draw_chessboard(frames[0], pattern_size, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
//                 }
//             }
//             catch (const std::runtime_error & e)
//             {
//                 std::cout << e.what() << '\n';
//             }
//         }
//         else if (modality_flags & USE_COLOR)
//         {
//             try
//             {          
//                 std::pair<std::pair<cv::Mat,cv::Mat>,timestamp_t> rs_peek = queue_rs.peek();
//                 frames[0] = rs_peek.first.first;
//                 if (find_patterns)
//                 {
//                     find_and_draw_chessboard(frames[0], pattern_size, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
//                 }
//             }
//             catch (const std::runtime_error & e)
//             {
//                 std::cerr << e.what() << '\n';
//             }
//         }
//         else if (modality_flags & USE_DEPTH)
//         {
//             try {
//                 std::pair<std::pair<cv::Mat,cv::Mat>,timestamp_t> rs_peek = queue_rs.peek();
//                 frames[1] = uls::DepthFrame::to_8bit(rs_peek.first.first, cv::COLORMAP_JET);
//             }
//             catch (const std::runtime_error & e)
//             {
//                 std::cerr << e.what() << '\n';
//             }
//         }

//         if (modality_flags & USE_THERM)
//         {
//             try
//             {
//                 std::pair<cv::Mat,timestamp_t> pt_peek = queue_pt.peek();
//                 frames[2] = uls::ThermalFrame::to_8bit(pt_peek.first);
//                 cv::cvtColor(frames[2], frames[2], cv::COLOR_GRAY2BGR);
//                 if (find_patterns)
//                     find_and_draw_chessboard(frames[2], pattern_size, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
//             }
//             catch (const std::runtime_error & e)
//             {
//                 std::cerr << e.what() << '\n';
//             }
//         }

//         // Error handling: before imshow, wait at least one frame captured for each active modality
//         if (
//             ((modality_flags & USE_COLOR > 0) && frames[0].empty()) ||
//             ((modality_flags & USE_DEPTH > 0) && frames[1].empty()) ||
//             ((modality_flags & USE_THERM > 0) && frames[2].empty()) 
//         )
//             continue;

//         cv::Mat tiling;
//         uls::tile(frames, 426, 240*frames.size(), 1, frames.size(), tiling);

//         cv::imshow(win_name, tiling);
//         cv::waitKey(1);
//     }
// }

int main(int argc, char * argv[]) try
{    
    /* -------------------------------- */
    /*   Command line argument parsing  */
    /* -------------------------------- */
    
    po::options_description desc("Program options");
    desc.add_options()
        ("help,h", "Print help messages")
        ("duration,d", po::value<int>()->default_value(8000), "Duration of the recording in milliseconds (ms)")
        ("modalities,M", po::value<std::string>()->default_value("color,depth,thermal"), "Comma-separated list of modalities to capture")
        ("find-pattern-on,F", po::value<std::string>()->implicit_value(""), "Comma-separated list of modalities to find pattern on (color and/or thermal)")
        ("pattern,p", po::value<std::string>()->default_value("8,9"), "Pattern size \"x,y\" squares")
        ("fps,f", po::value<int>()->default_value(30), "Acquisition speed (fps) of realsense (integer number 1~30)")
        ("verbosity,v", po::value<int>()->default_value(0), "Verbosity level (0: nothing | 1: countdown & output | 2: sections | 3: threads | 4: rs internals)")
        ("output-dir,o", po::value<std::string>()->default_value(""), "Output directory to save acquired data");
    
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
    /*    Main code    */
    /* --------------- */
        
    /* 
     * Set verbosity level:
     * 
     *   0: nothing
     *   1: countdown and output directory at the end of the program
     *   2: main program sections verbosity
     *   3: consumer/producer-level verbosity
     *   4: device-level verbosity
     */

    int verbosity = vm["verbosity"].as<int>();

    /* 
     * Define input modalities
     */

    int modality_flags = 0;

    std::vector<std::string> modalities;
    boost::split(modalities, vm["modalities"].as<std::string>(), boost::is_any_of(","));
    if (std::find(modalities.begin(), modalities.end(), "Color") != modalities.end()) 
        modality_flags |= USE_COLOR;
    if (std::find(modalities.begin(), modalities.end(), "Depth") != modalities.end()) 
        modality_flags |= USE_DEPTH;
    if (std::find(modalities.begin(), modalities.end(), "Thermal") != modalities.end())    
        modality_flags |= USE_THERM;
    
    assert(modality_flags > 0);

    /*
     * Define chessboard search in Color/Thermal modalities
     */

    int find_pattern_flags = 0;

    std::vector<std::string> find_pattern_on;
    if (vm.find("find-pattern-on") != vm.end())
    {
        boost::split(find_pattern_on, vm["find-pattern-on"].as<std::string>(), boost::is_any_of(","));
        if (std::find(find_pattern_on.begin(), find_pattern_on.end(), "Color") != find_pattern_on.end()) 
            find_pattern_flags |= FIND_PATTERN_ON_COLOR;
        if (std::find(find_pattern_on.begin(), find_pattern_on.end(), "Thermal") != find_pattern_on.end()) 
            find_pattern_flags |= FIND_PATTERN_ON_THERM;
    }

    /*
     * Define calibration pattern size
     */

    std::vector<std::string> pattern_dims;
    boost::split(pattern_dims, vm["pattern"].as<std::string>(), boost::is_any_of(","));
    assert(pattern_dims.size() == 2);
    
    int x = std::stoi(pattern_dims[0]);
    int y = std::stoi(pattern_dims[1]);
    assert(x > 2 && x > 2);

    cv::Size pattern_size (x,y);

    /*
     * Create output directory structures to store captured data
     */

    std::string date_and_time = current_time_and_date();

    fs::path parent (vm["output-dir"].as<std::string>());
    if (!parent.empty())
    {
        if (verbosity > 1) 
            std::cout << "[Main] Creating output directory structure ...\n";

        parent = parent / fs::path(date_and_time);
        if (modality_flags & USE_COLOR) 
            boost::filesystem::create_directories(parent / fs::path("rs/color/"));
        if (modality_flags & USE_DEPTH) 
            boost::filesystem::create_directories(parent / fs::path("rs/depth/"));
        if (modality_flags & USE_THERM) 
            boost::filesystem::create_directories(parent / fs::path("pt/thermal/"));

        if (verbosity > 1) 
            std::cout << "[Main] Output directory structure \"" << parent.string() << "\"created\n";
    }

    /*
     * Initialize devices
     */
    if (verbosity > 1) 
        std::cout << "[Main] Initializing devices ...\n";

    rs2::pipeline pipe_rs;
    pt::pipeline pipe_pt;
    if (verbosity > 4) 
        rs2::log_to_console(RS2_LOG_SEVERITY_DEBUG);
    
    rs2::config cfg;
    if (modality_flags & USE_COLOR)  
        cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, vm["fps"].as<int>());
    if (modality_flags & USE_DEPTH) 
        cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, vm["fps"].as<int>());
    // cfg.enable_stream(RS2_STREAM_INFRARED, 1280, 720, RS2_FORMAT_Y8, 15);
    
    rs2::pipeline_profile profile;
    if (modality_flags & (USE_COLOR | USE_DEPTH))
        profile = pipe_rs.start(cfg);
    if (modality_flags & USE_THERM)
        pipe_pt.start(); // no need for external configuration

    if (verbosity > 1) 
        std::cout << "[Main] Devices initialized ...\n";

    /*
     * Countdown and timing
     */

    int total_count = 3;
    int warn_every = 1;
    countdown(total_count, warn_every, verbosity > 0); // sleep for total_count (seconds) and warn every warn_every (seconds)

    // Timer thread will time the seconds to record (uses "is_capturing" boolean variable)
    bool is_capturing;
    std::thread timer_thr(timer, vm["duration"].as<int>(), std::ref(is_capturing)); // Timer will set is_capturing=false when finished

    while (!is_capturing) {
        std::cout << "[Main] Waiting timer to fire ...\n";
        continue; // safety check
    }

    /*
     * Initialize consumer-producer queues
     */

    std::thread p_rs_thr, p_pt_thr; // producers
    std::thread c_rs_thr, c_pt_thr; // consumers

    SafeQueue<rs_frame_t> queue_rs;
    SafeQueue<pt_frame_t> queue_pt;

    // Producer threads initialization

    if (verbosity > 1) std::cout << "[Main] Starting RS consumer/producer threads ...\n";
    // rs-related producers
    if (modality_flags & (USE_COLOR | USE_DEPTH))
        p_rs_thr = std::thread(produce_realsense, std::ref(pipe_rs), std::ref(profile), std::ref(queue_rs), std::ref(is_capturing), modality_flags, verbosity > 2);
    // else if (modality_flags & USE_COLOR)
    //     p_rs_thr = std::thread(produce_realsense_color, std::ref(pipe_rs), std::ref(profile), std::ref(queue_rs), std::ref(is_capturing), verbosity > 2);
    // else if (modality_flags & USE_DEPTH)
    //     p_rs_thr = std::thread(produce_realsense_depth, std::ref(pipe_rs), std::ref(profile), std::ref(queue_rs), std::ref(is_capturing), verbosity > 2);
    // pt producer
    if (modality_flags & USE_THERM)
        p_pt_thr = std::thread(produce_purethermal, std::ref(pipe_pt), std::ref(queue_pt), std::ref(is_capturing), verbosity > 2);

    if (verbosity > 1) std::cout << "[Main] Producer threads started ...\n";

    // Consumer threads initialization

    if (verbosity > 1) std::cout << "[Main] Starting consumer threads ...\n";
    // rs-related consumers
    if (modality_flags & (USE_COLOR | USE_DEPTH))
        c_rs_thr = std::thread(consume_realsense, std::ref(queue_rs), std::ref(is_capturing), parent, verbosity > 2);
    // else if (modality_flags & USE_COLOR)
    //     c_rs_thr = std::thread(consume_realsense_color, std::ref(queue_rs), std::ref(is_capturing), parent, verbosity > 2);
    // else if (modality_flags & USE_DEPTH)
    //     c_rs_thr = std::thread(consume_realsense_depth, std::ref(queue_rs), std::ref(is_capturing), parent, verbosity > 2);
    // pt consumer
    if (modality_flags & USE_THERM)
        c_pt_thr = std::thread(consume_purethermal, std::ref(queue_pt), std::ref(is_capturing), parent, verbosity > 2);

    if (verbosity > 1) 
        std::cout << "[Main] Consumer threads started ...\n";

    /* Visualization loop. imshow needs to be in the main thread! */

    if (verbosity > 1) 
        std::cout << "[Main] Starting visualization ...\n";

    /* 
     * Visualization loop
     * ------------------
     */
    visualize("Viewer", queue_rs, queue_pt, is_capturing, pattern_size, find_pattern_flags);

    if (verbosity > 1) 
        std::cout << "[Main] Visualization (and capturing) ended ...\n";

    /* Wait for consumer threads to finish */
    if (verbosity > 1) 
        std::cout << "[Main] Joining all threads ...\n";

    timer_thr.join();
    c_rs_thr.join();
    c_pt_thr.join();
    p_rs_thr.join();
    p_pt_thr.join();

    if (verbosity > 1) 
        std::cout << "[Main] All threads joined ...\n";

    if (verbosity > 1) 
        std::cout << "[Main] Stopping pipelines ...\n";

    pipe_rs.stop();
    pipe_pt.stop();

    if (verbosity > 1) 
        std::cout << "[Main] Pipelines stopped ...\n";

    if (verbosity > 0) 
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

