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
#include <boost/timer.hpp>

#include "pt_pipeline.hpp"
#include "safe_queue.hpp"
#include "utils.hpp"
#include "detection.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;

// Bit masks
#define USE_COLOR   static_cast<int>( 1 << 0 )
#define USE_DEPTH   static_cast<int>( 1 << 1 )
#define USE_THERM   static_cast<int>( 1 << 2 )

#define FIND_PATTERN_ON_COLOR   static_cast<int>( 1 << 0 )
#define FIND_PATTERN_ON_THERM   static_cast<int>( 1 << 1 )

// Error codes enum
namespace
{
    const size_t ERROR_UNHANDLED_EXCEPTION = -1;
    const size_t SUCCESS = 0;
    const size_t ERROR_IN_COMMAND_LINE = 1;
    const size_t ERROR_RS = 3;
    const size_t ERROR_PT = 4;
}

// Define structs

typedef struct
{
    std::atomic<bool> capture {true};
    std::atomic<bool> save_started {false};
    std::atomic<bool> save_suspended {true};
    std::chrono::duration<double,std::milli> capture_elapsed;
    std::chrono::duration<double,std::milli> save_elapsed;
} sync_flags_t;


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

typedef struct
{
    cv::Mat x;
    cv::Mat y;
    bool flip;
} map_t;

typedef struct
{
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
} intrinsics_t;

typedef struct
{
    cv::Mat R;
    cv::Mat T;
} extrinsics_t;


cv::Mat camera_matrix_pt = (cv::Mat_<double>(3,3) << 9.5045533560700073e+02, 0., 6.3047112789381094e+02, 0.,
       9.4565936858393161e+02, 3.5811018136651228e+02, 0., 0., 1. );
cv::Mat dist_coeffs_pt = (cv::Mat_<double>(5,1) <<  -3.3320320049230417e-01, 
                                                    2.0782685727252564e-01,
                                                    -4.5030959847842762e-03, 
                                                    8.0929155821229384e-04, 
                                                    4.4536256165454396e-02);

cv::Mat camera_matrix_rs = (cv::Mat_<double>(3,3) << 9.0165326877232167e+02, 0., 6.3855354574722332e+02, 0.,
       9.0029445566592142e+02, 3.7951370908973325e+02, 0., 0., 1.);
cv::Mat dist_coeffs_rs = (cv::Mat_<double>(5,1) <<  1.5528661227160781e-01, -5.0445540766661190e-01,
       4.2313291650837625e-03, 2.6515291487574665e-03, 4.5255308613010831e-01);

//
// FUNCTIONS
//

float get_depth_scale(rs2::device dev)
{
    for (rs2::sensor& sensor : dev.query_sensors())
    {
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
            return dpt.get_depth_scale();
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
 * Return current date string in a certain format: Y-m-d_H.M.S"
 * 
 * More details at: https://stackoverflow.com/questions/17223096/outputting-date-and-time-in-c-using-stdchrono
 * 
 * @return Current date string
 */
std::string current_time_and_date()
{
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H.%M.%S");
    return ss.str();
}

/*
 * Enqueues realsense's frames in a thread-safe queue while is_capturing flag is active.
 * 
 * @param pipe Is a user-created rs2::pipeline object
 * @param profile Is the active profile in pipe (returned when pipe is started)
 * @param q Is the safe-thread queue
 * @param is_capturing The boolean flag indicating if the function must keep enqueuing captured realsense frames
 * @param modality_flags Indicates the active modalities (check bit masks, namely USE_{COLOR,DEPTH,THERM}, defined in capture-rgbt.cpp)
 * @param verbose Prints enqueuing information
 * @return
 */
void produce_realsense(rs2::pipeline & pipe, rs2::pipeline_profile & profile, SafeQueue<rs_frame_t> & q, sync_flags_t & sync_flags, int modality_flags, bool verbose)
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
    while (sync_flags.capture)
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
            cv::Mat img_d = cv::Mat(cv::Size(aligned_depth_frame.get_width(), aligned_depth_frame.get_height()), 
                                  CV_16U, 
                                  const_cast<void *>(aligned_depth_frame.get_data()), 
                                  cv::Mat::AUTO_STEP);
            img_d.convertTo(img_d, CV_32F);
            frame.img_d = img_d * depth_scale;
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
            // frame.img_d = cv::Mat(cv::Size(depth_frame.get_width(), depth_frame.get_height()), 
            //                       CV_16U, 
            //                       const_cast<void *>(depth_frame.get_data()), 
            //                       cv::Mat::AUTO_STEP).clone();
            cv::Mat img_d = cv::Mat(cv::Size(depth_frame.get_width(), depth_frame.get_height()), 
                                  CV_16U, 
                                  const_cast<void *>(depth_frame.get_data()), 
                                  cv::Mat::AUTO_STEP);
            img_d.convertTo(img_d, CV_32F);
            frame.img_d = img_d * depth_scale;
        }

        q.enqueue( frame );
        if (verbose)
            std::cerr << "[RS] <PRODUCER> enqueued " << q.size() << " ..." << '\n';
    }

    if (p_align != NULL) 
        delete p_align;
}

/*
 * Enqueues purethermal's frames in a thread-safe queue while is_capturing flag is active.
 * 
 * @param pipe Is a user-created pt::pipeline
 * @param q Is the safe-thread queue
 * @param is_capturing The boolean flag indicating if the function must keep enqueuing captured realsense frames
 * @param verbose Prints enqueuing information
 * @return
 */
void produce_purethermal(pt::pipeline & p, SafeQueue<pt_frame_t> & q, sync_flags_t & sync_flags, bool verbose)
{
    while (sync_flags.capture)
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

/*
 * Dequeues realsense's frames from a thread-safe queue while is_capturing flag is active.
 * 
 * @param q Is the safe-thread queue
 * @param is_capturing The boolean flag indicating if the function must keep dequeueing captured realsense frames
 * @param dir Directory where the function dequeues and saves captured frames
 * @param verbose Prints enqueuing information
 * @return
 */
void consume_realsense(SafeQueue<rs_frame_t> & q, 
                       sync_flags_t & sync_flags, 
                       fs::path dir, 
                       uls::MovementDetector md, 
                       int duration, 
                       bool verbose)
{
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);
    
    int fid = 0;
    boost::format fmt("%08d");
    std::ofstream outfile;

    if (!dir.empty())
        outfile.open((dir / "rs.log").string(), std::ios_base::app);

    // auto start = std::chrono::steady_clock::now();
    // cv::Mat capture_img_prev;
    

    while (sync_flags.capture || q.size() > 0)
    {
        rs_frame_t capture = q.dequeue();
        if (verbose)
            std::cout << "[RS] >CONSUMER< dequeued " << q.size() << " ..." << '\n';

        // std::chrono::duration<double,std::milli> elapsed (std::chrono::steady_clock::now() - start);
        // if (!capture_img_prev.empty())
        // {
        //     cv::Mat capture_diff, gray_diff;
        //     cv::absdiff(capture.img_c, capture_img_prev, capture_diff);
        //     cv::cvtColor(capture_diff, gray_diff, cv::COLOR_BGR2GRAY);
        //     // do stuff
        //     double minVal, maxVal;
        //     int minIdx, maxIdx;
        //     cv::minMaxIdx(gray_diff, &minVal, &maxVal, &minIdx, &maxIdx);
        //     std::cout << minVal << " " << maxVal << std::endl;
        //     start = std::chrono::steady_clock::now();
        // }

        if (sync_flags.save_started && !sync_flags.save_suspended && !dir.empty())
        {
            long time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(capture.ts.time_since_epoch()).count();
            fmt % fid;
            outfile << fmt.str() << ',' << std::to_string(time_ms) << '\n';
        
            fs::path color_dir = dir / "rs/color/";
            fs::path depth_dir = dir / "rs/depth/";
            if (!capture.img_c.empty())
                cv::imwrite((color_dir / ("c_" + fmt.str() + ".jpg")).string(), capture.img_c.clone());
            if (!capture.img_d.empty())
                cv::imwrite((depth_dir / ("d_" + fmt.str() + ".png")).string(), capture.img_d.clone(), compression_params);
        }

        // capture_img_prev = capture.img_c;
        fid++;
    }
}

/*
 * Dequeues purethermal's frames from a thread-safe queue while is_capturing flag is active.
 * 
 * @param q Is the safe-thread queue
 * @param is_capturing The boolean flag indicating if the function must keep dequeueing captured realsense frames
 * @param dir Directory where the function dequeues and saves captured frames
 * @param verbose Prints enqueuing information
 * @return
 */
void consume_purethermal(SafeQueue<pt_frame_t> & q, 
                         sync_flags_t & sync_flags, 
                         fs::path dir, 
                         uls::MovementDetector md, 
                         int duration, 
                         bool verbose)
{
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);
    
    int fid = 0;
    boost::format fmt("%08d");
    std::ofstream outfile;
    
    if (!dir.empty())
        outfile.open((dir / "pt.log").string(), std::ios_base::app);
    
    auto start = std::chrono::steady_clock::now();
    cv::Mat capture_img_prev;

    while (sync_flags.capture || q.size() > 0)
    {
        pt_frame_t capture = q.dequeue();
        if (verbose)
            std::cout << "[PT] >CONSUMER< dequeued " << q.size() << " ..." << '\n';

        if (sync_flags.save_started)
        {
            if (duration <= 0)
            {
                sync_flags.save_suspended = false;
            }
            else
            {
                std::chrono::duration<double,std::milli> elapsed (std::chrono::steady_clock::now() - start);
                if ( elapsed >= std::chrono::milliseconds(duration) )
                {   
                    sync_flags.save_suspended = true;
                    if ( duration <= 0 || md.find(capture.img) ) // if movement detected
                    {
                        start = std::chrono::steady_clock::now(); // reset clock
                        sync_flags.save_suspended = false;
                    }
                }
            }

            if ( !sync_flags.save_suspended && !dir.empty())
            {
                long time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(capture.ts.time_since_epoch()).count();
                fmt % fid;
                outfile << fmt.str() << ',' << std::to_string(time_ms) << '\n';
                
                fs::path thermal_dir = dir / "pt/thermal/";
                cv::imwrite((thermal_dir / ("t_" + fmt.str() + ".png")).string(), capture.img, compression_params);
            }
            
        }

        fid++;
    }
}

/*
 * Timer function switching is_capturing from true to false after some time.
 * 
 * @param duration Number of millisecons after which is_capturing is put to false.
 * @param is_capturing The boolean flag indicating if the function must keep dequeueing captured realsense frames
 * @return
 */
void timer(int duration, std::atomic<bool> & b, std::chrono::duration<double,std::milli> & elapsed)
{
    auto start = std::chrono::steady_clock::now();

    // is_capturing = true;
    elapsed = std::chrono::duration<double,std::milli>(0);
    while (elapsed < std::chrono::milliseconds(duration))
    {
        elapsed = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - start);
    }
    b = !b;
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

/*
 * Counts down "total" time and warns every "warn_every" time instants
 * 
 * @param total Total time to count down
 * @param warn_every After every "warn_every" time instants, print a message with the countdown
 * @param verbose Prints enqueuing information
 * @return
 */
template<typename T>
void countdown(int total, int warn_every, bool verbose = true)
{
    while (total > 0) 
    {
        if (verbose) std::cout << total << '\n';
        std::this_thread::sleep_for(T(warn_every));
        total -= warn_every;
    }
}

void find_and_draw_chessboard(cv::Mat & img, cv::Size pattern_size, int flags = 0)
{
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::Mat corners;
    bool found = cv::findChessboardCorners(gray, pattern_size, corners, flags);
    
    if (img.type() != CV_8UC3)
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

    cv::drawChessboardCorners(img, pattern_size, corners, found);
}


// void visualize(std::string win_name,
//                SafeQueue<rs_frame_t> & queue_rs, 
//                SafeQueue<pt_frame_t> & queue_pt, 
//                sync_flags_t & sync_flags, 
//                std::chrono::duration<double,std::milli> & capture_elapsed,
//                cv::Size pattern_size, 
//                int find_pattern_flags = 0,
//                map_t* map_rs = NULL,
//                map_t* map_pt = NULL)
// {
//     // If peek fails for some stream, the previous frame is shown.
//     std::vector<cv::Mat> frames (3); // color, depth, thermal

//     // int fid = 0;
//     // boost::format fmt("%08d");

//     while (sync_flags.capture)
//     {
//         rs_frame_t rs_frame;
//         pt_frame_t pt_frame;
//         try 
//         {
//             rs_frame = queue_rs.peek();
//             if (!rs_frame.img_c.empty())
//             {
//                 frames[0] = rs_frame.img_c.clone();
//                 // uls::resize(frames[0], frames[0], cv::Size(map_rs->x.cols, map_rs->x.rows));
//                 if (map_rs != NULL)
//                 {
//                     cv::remap(frames[0], frames[0], map_rs->x, map_rs->y, cv::INTER_LINEAR);
//                     if (map_rs->flip) cv::flip(frames[0], frames[0], 0);
//                 }
//                 if (find_pattern_flags & FIND_PATTERN_ON_COLOR) 
//                 {
//                     // cv::resize(frames[0],frames[0],cv::Size(640,360));
//                     find_and_draw_chessboard(frames[0], pattern_size, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE/* + cv::CALIB_CB_FAST_CHECK + cv::CALIB_CB_FAST_CHECK*/);
//                 }
//             }
//             if (!rs_frame.img_d.empty())
//             {
//                 cv::Mat tmp = uls::DepthFrame::cut_at(rs_frame.img_d.clone(), 4000);
//                 frames[1] = uls::DepthFrame::to_8bit(tmp, cv::COLORMAP_BONE);
//                 // uls::resize(frames[1], frames[1], cv::Size(map_rs->x.cols, map_rs->x.rows));
//                 if (map_rs != NULL && !rs_frame.img_c.empty())
//                 {
//                     cv::remap(frames[1], frames[1], map_rs->x, map_rs->y, cv::INTER_LINEAR);
//                     if (map_rs->flip) cv::flip(frames[1], frames[1], 0);
//                 }
//             }
//         }
//         catch (const std::runtime_error & e) // catch peek's exception
//         {
//             std::cout << e.what() << '\n';
//         }
        
//         try
//         {
//             pt_frame = queue_pt.peek();
//             if (!pt_frame.img.empty())
//             {
//                 frames[2] = uls::ThermalFrame::to_8bit(pt_frame.img.clone());
//                 uls::resize(frames[2], frames[2], cv::Size(1280, 720));
//                 if (map_pt != NULL)
//                 {
//                     cv::remap(frames[2], frames[2], map_pt->x, map_pt->y, cv::INTER_LINEAR);
//                     if (map_pt->flip) cv::flip(frames[2], frames[2], 0);
//                 }
//                 if (find_pattern_flags & FIND_PATTERN_ON_THERM)
//                 {
//                     cv::cvtColor(frames[2], frames[2], cv::COLOR_GRAY2BGR);
//                     find_and_draw_chessboard(frames[2], pattern_size, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
//                 }
//             }
//         }
//         catch (const std::runtime_error & e) // catch peek's exception
//         {
//             std::cerr << e.what() << '\n';
//         }

//         // int64_t time_rs, time_pt;
//         // time_rs = std::chrono::duration_cast<std::chrono::milliseconds>(rs_frame.ts.time_since_epoch()).count();
//         // time_pt = std::chrono::duration_cast<std::chrono::milliseconds>(pt_frame.ts.time_since_epoch()).count();
//         // std::cout << time_rs - time_pt << '\n';
//         // if (time_rs > 0 && time_pt > 0 && std::abs(time_rs - time_pt) < 500)
//         // {
//             // std::vector<cv::Mat> frames_gray (3);
//             // for (int i = 0; i < frames.size()-1; i++)
//             // {
//             //     if (frames[i].empty())
//             //         frames_gray[i].create(1280, 720, CV_8UC1);
//             //     else if (frames[i].type() == CV_8UC3)
//             //         cv::cvtColor(frames[i], frames_gray[i], cv::COLOR_BGR2GRAY);
//             //     else
//             //         frames[i].copyTo(frames_gray[i]);
//             // }
//             // cv::merge(frames_gray, frames[3]);
//             // std::cout << frames[3].channels() << ',' << frames[3].type() << '\n';

//             // std::vector<cv::Mat> subset = {frames[1], frames[2]};
//             cv::Mat tiling;
//             uls::tile(frames, 534, 900, 1, 3, tiling);

//             //fmt % (fid++);
//             // cv::imwrite("./demo/" + fmt.str() + ".jpg", tiling);
        
//             tiling = (sync_flags.save_started && !sync_flags.save_suspended) ? tiling : ~tiling;

//             std::stringstream ss;
//             ss << capture_elapsed.count(); 
//             cv::putText(tiling, ss.str(), cv::Point(tiling.cols/20.0,tiling.rows/20.0), CV_FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,0,255), 1, 8, false);

//             cv::imshow(win_name, tiling);
//             cv::waitKey(1);
//         // }
//     }
// }

void get_intrinsic_parameters(cv::Mat K, double & fx, double & fy, double & cx, double & cy)
{
    fx = K.at<double>(0,0);
    fy = K.at<double>(1,1);
    cx = K.at<double>(0,2);
    cy = K.at<double>(1,2);
}

void align_to_depth(cv::Mat depth, 
                    cv::Mat K_depth,
                    cv::Mat K_other,
                    std::shared_ptr<extrinsics_t> extrinsics,
                    cv::Mat & map_x, 
                    cv::Mat & map_y)
{
    double fx_d, fy_d, cx_d, cy_d, fx_o, fy_o, cx_o, cy_o;
    get_intrinsic_parameters(K_depth, fx_d, fy_d, cx_d, cy_d);
    get_intrinsic_parameters(K_other, fx_o, fy_o, cx_o, cy_o);

    cv::Mat R = extrinsics->R;
    cv::Mat T = extrinsics->T;

    double x, y, z;
    double p_x, p_y, p_z;

    map_x.release();
    map_y.release();

    map_x.create(depth.size(), CV_32FC1);
    map_y.create(depth.size(), CV_32FC1);

    for (int i = 0; i < depth.rows; i++)
    {
        for (int j = 0; j < depth.cols; j++)
        {
            z = depth.at<float>(i,j);
            if (z > 0)
            {
                x = (j - cx_d) * z / fx_d;
                y = (i - cy_d) * z / fy_d;

                // cv::Mat p = (cv::Mat_<double>(3,1) << x, y, z);
                // cv::Mat pp = extrinsics->R * p + extrinsics->T;
                // double pp_x, pp_y, pp_z;
                // pp_x = pp.at<double>(0,0);
                // pp_y = pp.at<double>(1,0);
                // pp_z = pp.at<double>(2,0);

                p_x = (R.at<double>(0,0) * x + R.at<double>(0,1) * y + R.at<double>(0,2) * z) + T.at<double>(0,0);
                p_y = (R.at<double>(1,0) * x + R.at<double>(1,1) * y + R.at<double>(1,2) * z) + T.at<double>(1,0);
                p_z = (R.at<double>(2,0) * x + R.at<double>(2,1) * y + R.at<double>(2,2) * z) + T.at<double>(2,0);

                // double xx = (p_x * fx_o / p_z) + cx_o;
                // double yy = (p_y * fy_o / p_z) + cy_o;

                map_x.at<float>(i,j) = (p_x * fx_o / p_z) + cx_o;
                map_y.at<float>(i,j) = (p_y * fy_o / p_z) + cy_o;
            }
        }
    }

}

void visualize(std::string win_name,
               SafeQueue<rs_frame_t> & queue_rs, 
               SafeQueue<pt_frame_t> & queue_pt, 
               std::map<std::string, intrinsics_t> intrinsics,
               int modality_flags,
               sync_flags_t & sync_flags, 
            //    std::chrono::duration<double,std::milli> & capture_elapsed,
               std::shared_ptr<extrinsics_t> extrinsics = NULL,
               int find_pattern_flags = 0,
               cv::Size pattern_size = cv::Size(6,5)
)
{
    // If peek fails for some stream, the previous frame is shown.
    // double fx_rs, fy_rs, cx_rs, cy_rs,
    //        fx_pt, fx_pt, fx_pt, fx_pt;
    // get_intrinsic_parameters(intrinsics_rs->camera_matrix, fx_rs, fy_rs, cx_rs, cy_rs);
    // get_intrinsic_parameters(intrinsics_pt->camera_matrix, fx_pt, fx_pt, fx_pt, fx_pt);

    bool use_depth = (modality_flags & USE_DEPTH) > 0;

    std::vector<cv::Mat> frames (3);

    while (sync_flags.capture)
    {
        rs_frame_t rs_frame;
        pt_frame_t pt_frame;

        cv::Mat depth;
        try 
        {
            rs_frame = queue_rs.peek(33);
            if (!rs_frame.img_c.empty())
            {
                cv::Mat color (rs_frame.img_c.size(), rs_frame.img_c.type());
                cv::undistort(rs_frame.img_c, color, intrinsics["Color"].camera_matrix, intrinsics["Color"].dist_coeffs);

                if (find_pattern_flags & FIND_PATTERN_ON_COLOR) 
                    find_and_draw_chessboard(color, pattern_size, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE/* + cv::CALIB_CB_FAST_CHECK + cv::CALIB_CB_FAST_CHECK*/);

                frames[0] = color;
            }

            if (!rs_frame.img_d.empty())
            {
                // depth = uls::DepthFrame::cut_at<float>(rs_frame.img_d.clone(), 4000, 0.f);
                depth.create(rs_frame.img_d.size(), rs_frame.img_d.type());

                if (rs_frame.img_c.empty()) // rs's intrinsics are computed from color stream -> undistort depth if aligned to color
                    cv::undistort(rs_frame.img_d, depth, intrinsics["Color"].camera_matrix, intrinsics["Color"].dist_coeffs);
                else
                    depth = rs_frame.img_d;

                frames[1] = uls::DepthFrame::to_8bit(depth, cv::COLORMAP_BONE);
            }
        }
        catch (const std::runtime_error & e) // catch peek's exception
        {
            std::cout << e.what() << '\n';
        }
        
        if ( !use_depth || (extrinsics && use_depth && !depth.empty()) )
        {
            try
            {
                pt_frame = queue_pt.peek(33);
                if (!pt_frame.img.empty())
                {
                    cv::Mat therm = uls::ThermalFrame::to_8bit(pt_frame.img.clone());
                    uls::resize(therm, therm, cv::Size(1280,720));

                    cv::Mat tmp = therm.clone();
                    cv::undistort(tmp, therm, intrinsics["Thermal"].camera_matrix, intrinsics["Thermal"].dist_coeffs);

                    if (extrinsics && use_depth && !depth.empty())
                    {
                        cv::Mat map_x, map_y;
                        align_to_depth(depth, intrinsics["Color"].camera_matrix, intrinsics["Thermal"].camera_matrix, extrinsics, map_x, map_y);
                        // // cv::Mat therm_aligned (720,1280,CV_8UC1);
                        // cv::Mat map_x (720, 1280, CV_32FC1);
                        // cv::Mat map_y (720, 1280, CV_32FC1);
                        // // therm_aligned.setTo(0);
                        // // map_x.setTo(-1);
                        // // map_y.setTo(-1);
                        // for (int i = 0; i < depth.rows; i++)
                        // {
                        //     for (int j = 0; j < depth.cols; j++)
                        //     {
                        //         double x, y, z;
                        //         z = depth.at<float>(i,j);
                        //         if (z > 0)
                        //         {
                        //             x = (j - cx_rs) * z / fx_rs;
                        //             y = (i - cy_rs) * z / fy_rs;

                        //             cv::Mat p = (cv::Mat_<double>(3,1) << x, y, z);
                        //             cv::Mat pp = extrinsics->R * p + extrinsics->T;
                        //             double pp_x, pp_y, pp_z;
                        //             pp_x = pp.at<double>(0,0);
                        //             pp_y = pp.at<double>(1,0);
                        //             pp_z = pp.at<double>(2,0);
                        //             double xx = (pp_x * fx_pt / pp_z) + cx_pt;
                        //             double yy = (pp_y * fy_pt / pp_z) + cy_pt;
                        //             map_x.at<float>(i,j) = xx;
                        //             map_y.at<float>(i,j) = yy;
                        //             // if (xx < 1280 && xx > 0 && yy < 720 && yy > 0)
                        //             //     therm_aligned.at<unsigned char>(i,j) = frames[2].at<unsigned char>(yy,xx);
                                    
                        //         }
                        //     }
                        // }
                        // // therm_aligned.copyTo(therm);
                        cv::remap(therm, therm, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
                    }
                    
                    if (find_pattern_flags & FIND_PATTERN_ON_THERM)
                    {
                        cv::cvtColor(therm, therm, cv::COLOR_GRAY2BGR);
                        find_and_draw_chessboard(therm, pattern_size, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
                    }

                    frames[2] = therm;
                }
            }
            catch (const std::runtime_error & e) // catch peek's exception
            {
                std::cerr << e.what() << '\n';
            }
        }

        // int64_t time_rs, time_pt;
        // time_rs = std::chrono::duration_cast<std::chrono::milliseconds>(rs_frame.ts.time_since_epoch()).count();
        // time_pt = std::chrono::duration_cast<std::chrono::milliseconds>(pt_frame.ts.time_since_epoch()).count();
        // std::cout << time_rs - time_pt << '\n';
        // if (time_rs > 0 && time_pt > 0 && std::abs(time_rs - time_pt) < 500)
        // {
            // std::vector<cv::Mat> frames_gray (frames.size()-1);
            // for (int i = 0; i < frames.size()-1; i++)
            // {
            //     if (frames[i].empty())
            //         frames_gray[i].create(1280, 720, CV_8UC1);
            //     else if (frames[i].type() == CV_8UC3)
            //         cv::cvtColor(frames[i], frames_gray[i], cv::COLOR_BGR2GRAY);
            //     else
            //         frames[i].copyTo(frames_gray[i]);
            // }
            // cv::merge(frames_gray, frames[frames.size()-1]);

            cv::Mat tiling;
            uls::tile(frames, 534, 900, 1, frames.size(), tiling);
        
            tiling = (sync_flags.save_started && !sync_flags.save_suspended) ? tiling : ~tiling;

            std::stringstream ss;
            ss << sync_flags.capture_elapsed.count(); 
            cv::putText(tiling, ss.str(), cv::Point(tiling.cols/20.0,tiling.rows/20.0), CV_FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,0,255), 1, 8, false);

            cv::imshow(win_name, tiling);
            cv::waitKey(1);
        // }
    }
}

//
// MAIN CODE
//

int main(int argc, char * argv[]) try
{    
    /* -------------------------------- */
    /*   Command line argument parsing  */
    /* -------------------------------- */

    bool mov_ec = false;

    po::options_description desc("Program options");
    desc.add_options()
        ("help,h", "Print help messages")
        ("duration,d", po::value<int>()->default_value(8000), "Duration of the recording in milliseconds (ms)")
        ("modalities,M", po::value<std::string>()->default_value("color,depth,thermal"), "Comma-separated list of modalities to capture")
        ("find-pattern-on,F", po::value<std::string>()->implicit_value(""), "Comma-separated list of modalities to find pattern on (color and/or thermal)")
        ("pattern,p", po::value<std::string>()->default_value("8,7"), "Pattern size \"x,y\" squares")
        ("fps,f", po::value<int>()->default_value(30), "Acquisition speed (fps) of realsense (integer number 1~30)")
        ("calibration-params", po::value<std::string>()->default_value(""), "Calibration mapping parameters")
        ("md-duration", po::value<int>(), "When movement detected record during X millisecs")
        ("md-pixel-thresh", po::value<int>()->default_value(30), "When movement detected record during X millisecs")
        ("md-frame-ratio", po::value<float>()->default_value(0.01), "When movement detected record during X millisecs")
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
        
    // TODOs:
    // + Control ALL passed arguments:
    // - modalities: OK
    // - find-pattern-on: OK
    // - pattern: OK
    // - calibration-params: OK

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

    if (verbosity > 4) 
        rs2::log_to_console(RS2_LOG_SEVERITY_DEBUG);
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
     *///

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

    // bool calibrate = false;
    std::map<std::string, intrinsics_t> intrinsics;
    std::shared_ptr<extrinsics_t> extrinsics;
    if (!vm["calibration-params"].as<std::string>().empty())
    {
        cv::FileStorage fs (vm["calibration-params"].as<std::string>(), cv::FileStorage::READ);
        if (fs.isOpened())
        {
            std::string modality_1, modality_2;
            fs["modality-1"] >> modality_1;
            fs["modality-2"] >> modality_2;
           
            fs["camera_matrix_1"] >> intrinsics[modality_1].camera_matrix;
            fs["camera_matrix_2"] >> intrinsics[modality_2].camera_matrix;
            fs["dist_coeffs_1"]   >> intrinsics[modality_1].dist_coeffs;
            fs["dist_coeffs_2"]   >> intrinsics[modality_2].dist_coeffs;

            extrinsics = std::make_shared<extrinsics_t>();
            fs["R"] >> extrinsics->R;
            fs["T"] >> extrinsics->T;
            // calibrate = true;
        }
    }
    // std::string mapx_str ("mapx"), mapy_str ("mapy");
    // map_t map_rs, map_pt;
    // bool calibrate = false;
    // if (!vm["calibration-params"].as<std::string>().empty())
    // {
    //     cv::FileStorage fs (vm["calibration-params"].as<std::string>(), cv::FileStorage::READ);
    //     if (fs.isOpened())
    //     {
    //         std::string m1, m2;
    //         fs["modality-1"] >> m1;
    //         fs["modality-2"] >> m2;

    //         if (m1 == "Color")
    //         {
    //             fs[mapx_str + "-1"] >> map_rs.x;
    //             fs[mapy_str + "-1"] >> map_rs.y;
    //         }
    //         else if (m1 == "Thermal")
    //         {
    //             fs[mapx_str + "-1"] >> map_pt.x;
    //             fs[mapy_str + "-1"] >> map_pt.y;
    //         }

    //         if (m2 == "Color")
    //         {
    //             fs[mapx_str + "-2"] >> map_rs.x;
    //             fs[mapy_str + "-2"] >> map_rs.y;
    //         }
    //         else if (m2 == "Thermal")
    //         {
    //             fs[mapx_str + "-2"] >> map_pt.x;
    //             fs[mapy_str + "-2"] >> map_pt.y;
    //         }

    //         bool flip;
    //         fs["vflip"] >> flip;
    //         map_rs.flip = map_pt.flip = flip;

    //         calibrate = true;
    //     }
    // }

    /*
     * Initialize devices
     */
    if (verbosity > 1) 
        std::cout << "[Main] Initializing devices ...\n";

    // RealSense pipeline (config and start)
    rs2::pipeline pipe_rs;
    rs2::pipeline_profile profile;
    
    rs2::config cfg_rs;
    if (modality_flags & USE_COLOR)  
        cfg_rs.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, vm["fps"].as<int>());
    if (modality_flags & USE_DEPTH) 
        cfg_rs.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, vm["fps"].as<int>());
    
    if (modality_flags & (USE_COLOR | USE_DEPTH))
        profile = pipe_rs.start(cfg_rs);

    // PureThermal pipeline (config and start)
    pt::pipeline pipe_pt;
    pt::config cfg_pt;
    cfg_pt.set_stream(160, 120);

    if (modality_flags & USE_THERM)
        pipe_pt.start(cfg_pt);

    if (verbosity > 1) 
        std::cout << "[Main] Devices initialized ...\n";

    // Countdown
    countdown<std::chrono::seconds>(2, 1, verbosity > 3); // sleep for 3 second and print a message every 1 second

    // Timer thread will time the seconds to record (uses "is_capturing" boolean variable)
    // std::atomic<bool> is_capturing (true);
    // std::atomic<bool> is_saving (false);
    sync_flags_t sync_flags;
    // sync_flags.capture = true;
    // sync_flags.save_started = false;
    // sync_flags.save_suspended = true;

    // while (!is_capturing) {
    //     std::cout << "[Main] Waiting timer to fire ...\n";
    //     continue; // safety check
    // }

    /*
     * Initialize consumer-producer queues
     */

    std::thread p_rs_thr, p_pt_thr; // producers
    std::thread c_rs_thr, c_pt_thr; // consumers

    SafeQueue<rs_frame_t> queue_rs (1); // set minimum buffer size to 1 (to be able to peek frames in visualize function)
    SafeQueue<pt_frame_t> queue_pt (1);

    // Producer threads initialization

    if (verbosity > 1) std::cout << "[Main] Starting RS consumer/producer threads ...\n";
    // rs-related producers
    if (modality_flags & (USE_COLOR | USE_DEPTH))
        p_rs_thr = std::thread(produce_realsense, std::ref(pipe_rs), std::ref(profile), std::ref(queue_rs), std::ref(sync_flags), modality_flags, verbosity > 2);
    if (modality_flags & USE_THERM)
        p_pt_thr = std::thread(produce_purethermal, std::ref(pipe_pt), std::ref(queue_pt), std::ref(sync_flags), verbosity > 2);

    if (verbosity > 1) std::cout << "[Main] Producer threads started ...\n";

    // Consumer threads initialization

    uls::MovementDetector md (vm["md-pixel-thresh"].as<int>(), vm["md-frame-ratio"].as<float>());

    if (verbosity > 1) std::cout << "[Main] Starting consumer threads ...\n";
    // rs-related consumers
    if (modality_flags & (USE_COLOR | USE_DEPTH))
        c_rs_thr = std::thread(consume_realsense, std::ref(queue_rs), std::ref(sync_flags), 
                               parent, md, vm["md-duration"].as<int>(), verbosity > 2);
    if (modality_flags & USE_THERM)
        c_pt_thr = std::thread(consume_purethermal, std::ref(queue_pt), std::ref(sync_flags), 
                               parent, md, vm["md-duration"].as<int>(), verbosity > 2);

    if (verbosity > 1) 
        std::cout << "[Main] Consumer threads started ...\n";

    /* Visualization loop. imshow needs to be in the main thread! */

    if (verbosity > 1) 
        std::cout << "[Main] Starting visualization ...\n";

    /* 
     * Visualization loop
     * ------------------
     */

    // Open visualization window
    cv::namedWindow("Viewer");
    std::chrono::duration<double,std::milli> capture_elapsed, saving_elapsed;
    std::thread save_start_timer(timer, 3000, std::ref(sync_flags.save_started), std::ref(sync_flags.save_elapsed));
    std::thread capture_timer(timer, 3000 + vm["duration"].as<int>(), std::ref(sync_flags.capture), std::ref(sync_flags.capture_elapsed)); // Timer will set is_capturing=false when finished
    // visualize("Viewer", queue_rs, queue_pt, sync_flags, capture_elapsed, pattern_size, find_pattern_flags, calibrate ? &map_rs : NULL, calibrate ? &map_pt : NULL);
    visualize("Viewer", queue_rs, queue_pt, intrinsics, modality_flags, sync_flags, extrinsics, find_pattern_flags, pattern_size);
    cv::destroyWindow("Viewer");

    if (verbosity > 1) 
        std::cout << "[Main] Visualization (and capturing) ended ...\n";

    /* Wait for consumer threads to finish */
    if (verbosity > 1) 
        std::cout << "[Main] Joining all threads ...\n";

    save_start_timer.join();
    capture_timer.join();
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

