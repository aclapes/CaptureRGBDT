// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <iostream>
#include <librealsense2/rs.hpp>     // Include RealSense Cross Platform API
#include <librealsense2/rs_advanced_mode.hpp>
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

#include "pt/pipeline.hpp" // rs-style pipeline for purethermal device (pt)
#include "rs/helpers.hpp" // rs-related helper functions

#include "utils/safe_queue.hpp"
#include "utils/common.hpp"
#include "utils/calibration.hpp"
#include "utils/detection.hpp"

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

struct synchronization_t
{
    std::atomic<bool> capture {true};
    std::atomic<bool> save_started {false};
    std::atomic<bool> save_suspended {true};
    std::chrono::duration<double,std::milli> capture_elapsed;
    std::chrono::duration<double,std::milli> save_elapsed;
};

struct rs_frame_t
{
    cv::Mat img_c; // color
    cv::Mat img_d; // depth
    // cv::Mat img_ir; // ir
    std::chrono::high_resolution_clock::time_point ts;
};

struct pt_frame_t
{
    cv::Mat img; // thermal
    std::chrono::high_resolution_clock::time_point ts;
};

struct frame_buffer_t : public std::mutex
{
    rs_frame_t rs;
    pt_frame_t pt;
};

/*
 * The producer feeds a queue with multimodal frames from a RealSense and PureThermal device.
 * Producer function to be encapsulated in a thread (running in parallel to a consumer-encapsulated thread).
 * 
 * It keeps producing until the synchronization object tells it to stop.
 * 
 * @param pipe_rs Is a user-created rs2::pipeline object (provided by RealSense's lib)
 * @param pipe_pt Is a user-created pt::pipeline object (custom class to provide a common interface with rs2::pipeline)
 * @param profile Is the active profile in pipe (returned when pipe is started)
 * @param q Is a safe thred queue of multi-modal frames
 * @param sync Synchronization flags among threaded functions: producer, consumer, and visualizer.
 * @param modality_flags Indicates the active modalities (check bit masks, namely USE_{COLOR,DEPTH,THERM}, defined in capture-rgbt.cpp)
 * @param verbose Prints enqueuing information
 */
void produce(rs2::pipeline & pipe_rs, 
             pt::pipeline & pipe_pt,
             rs2::pipeline_profile & profile, 
             uls::SafeQueue< std::pair<rs_frame_t, pt_frame_t> > & q, 
             synchronization_t & sync, 
             int modality_flags, 
             bool verbose)
{

    rs2_stream align_to;
    rs2::align* p_align = nullptr;

    if ((modality_flags & USE_COLOR) > 0 && (modality_flags & USE_DEPTH) > 0)
    {
        align_to = rs_help::find_stream_to_align(profile.get_streams());
        p_align = new rs2::align(align_to);
    }
    
    while (sync.capture)
    {
        rs_frame_t frame_rs;
        pt_frame_t frame_pt;

        uvc_frame_t *img = pipe_pt.wait_for_frames(); // blocking instruction
        frame_pt.ts = std::chrono::high_resolution_clock::now();

        rs2::frameset frames = pipe_rs.wait_for_frames(); // blocking instruction
        frame_rs.ts = std::chrono::high_resolution_clock::now();

        if (modality_flags & USE_THERM)
        {
            frame_pt.img = cv::Mat(cv::Size(160, 120), CV_16UC1, img->data).clone();
        }

        if (((modality_flags & USE_COLOR) | (modality_flags & USE_DEPTH)) == (USE_COLOR | USE_DEPTH)) 
        {
            // Get processed aligned frame
            rs2::frameset processed = p_align->process(frames);
            // Trying to get both color and aligned depth frames
            rs2::video_frame other_frame = processed.first_or_default(align_to);
            rs2::depth_frame aligned_depth_frame = processed.get_depth_frame();
            // Copy color and depth rs_frame_t struct
            frame_rs.img_c = cv::Mat(cv::Size(other_frame.get_width(), other_frame.get_height()), 
                                  CV_8UC3, 
                                  const_cast<void *>(other_frame.get_data()), 
                                  cv::Mat::AUTO_STEP).clone();
            frame_rs.img_d = cv::Mat(cv::Size(aligned_depth_frame.get_width(), aligned_depth_frame.get_height()), 
                                  CV_16U, 
                                  const_cast<void *>(aligned_depth_frame.get_data()), 
                                  cv::Mat::AUTO_STEP).clone();
        }
        else if (modality_flags & USE_COLOR)
        {
            // Trying to get both color and aligned depth frames
            rs2::video_frame color_frame = frames.get_color_frame();
            // Copy color to rs_frame_t struct
            frame_rs.img_c = cv::Mat(cv::Size(color_frame.get_width(), color_frame.get_height()), 
                                  CV_8UC3, 
                                  const_cast<void *>(color_frame.get_data()), 
                                  cv::Mat::AUTO_STEP).clone();
        }
        else if (modality_flags & USE_DEPTH)
        {
            // Trying to get both color and aligned depth frames
            rs2::depth_frame depth_frame = frames.get_depth_frame();
            // Copy depth to rs_frame_t struct
            frame_rs.img_d = cv::Mat(cv::Size(depth_frame.get_width(), depth_frame.get_height()), 
                                  CV_16U, 
                                  const_cast<void *>(depth_frame.get_data()), 
                                  cv::Mat::AUTO_STEP).clone();
        }

        q.enqueue( std::pair<rs_frame_t, pt_frame_t>(frame_rs, frame_pt) );
        if (verbose)
            std::cerr << "[RS+PT] PRODUCER E<< " << q.size() << " ..." << '\n';
    }

    if (p_align) delete p_align;
}

/*
 * Enqueues realsense's frames in a thread-safe queue.
 * 
 * It keeps producing until the synchronization object tells it to stop.
 * 
 * @param pipe Is a user-created rs2::pipeline object
 * @param profile Is the active profile in pipe (returned when pipe is started)
 * @param q Is the safe-thread queue
 * @param sync Synchronization flags among threaded functions: producer, consumer, and visualizer.
 * @param modality_flags Indicates the active modalities (check bit masks, namely USE_{COLOR,DEPTH,THERM}, defined in capture-rgbt.cpp)
 * @param verbose Prints enqueuing information
 * @return
 */
void produce_realsense(rs2::pipeline & pipe, 
                       rs2::pipeline_profile & profile, 
                       uls::SafeQueue<rs_frame_t> & q, 
                       synchronization_t & sync, 
                       int modality_flags, 
                       bool verbose)
{
    rs2_stream align_to;
    rs2::align* p_align = nullptr;

    if ((modality_flags & USE_COLOR) > 0 && (modality_flags & USE_DEPTH) > 0)
    {
        align_to = rs_help::find_stream_to_align(profile.get_streams());
        p_align = new rs2::align(align_to);
    }
    
    while (sync.capture)
    {
        rs2::frameset frames = pipe.wait_for_frames(); // blocking instruction

        rs_frame_t frame;
        frame.ts = std::chrono::high_resolution_clock::now();
        
        if (((modality_flags & USE_COLOR) | (modality_flags & USE_DEPTH)) == (USE_COLOR | USE_DEPTH)) 
        {
            // Get processed aligned frame
            rs2::frameset processed = p_align->process(frames);
            // Trying to get both color and aligned depth frames
            rs2::video_frame other_frame = processed.first_or_default(align_to);
            rs2::depth_frame aligned_depth_frame = processed.get_depth_frame();
            // Copy color and depth rs_frame_t struct
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
            // Copy color to rs_frame_t struct
            frame.img_c = cv::Mat(cv::Size(color_frame.get_width(), color_frame.get_height()), 
                                  CV_8UC3, 
                                  const_cast<void *>(color_frame.get_data()), 
                                  cv::Mat::AUTO_STEP).clone();
        }
        else if (modality_flags & USE_DEPTH)
        {
            // Trying to get both color and aligned depth frames
            rs2::depth_frame depth_frame = frames.get_depth_frame();
            // Copy depth to rs_frame_t struct
            frame.img_d = cv::Mat(cv::Size(depth_frame.get_width(), depth_frame.get_height()), 
                                  CV_16U, 
                                  const_cast<void *>(depth_frame.get_data()), 
                                  cv::Mat::AUTO_STEP).clone();
        }

        q.enqueue( frame );
        if (verbose)
            std::cerr << "[RS] PRODUCER E<< " << q.size() << " ..." << '\n';
    }

    if (p_align) delete p_align;
}

/*
 * Enqueues purethermal's frames in a thread-safe queue.
 * 
 * It keeps producing until the synchronization object tells it to stop.
 *
 * @param pipe Is a user-created pt::pipeline
 * @param q Is the safe-thread queue
 * @param sync Synchronization flags among threaded functions: producer, consumer, and visualizer.
 * @param verbose Prints enqueuing information
 * @return
 */
void produce_purethermal(pt::pipeline & p, 
                         uls::SafeQueue<pt_frame_t> & q, 
                         synchronization_t & sync, 
                         bool verbose)
{
    std::chrono::high_resolution_clock::time_point t1, t2;
    while (sync.capture)
    {
        uvc_frame_t *img = p.wait_for_frames(); // blocking instruction

        pt_frame_t frame;
        frame.ts = std::chrono::high_resolution_clock::now();
        frame.img = cv::Mat(cv::Size(160, 120), CV_16UC1, img->data).clone();
        
        q.enqueue( frame );

        if (verbose)
            std::cerr << "[PT] PRODUCER E<< " << q.size() << " ..." << '\n';
    }
}

/*
 * Dequeues realsense's frames from a thread-safe queue.
 * 
 * It keeps consuming until the synchronization object tells it to stop.
 * 
 * @param q Is the safe-thread queue of RealSense's frames
 * @param last_frame Keep a buffer of the last consumed frame for the visualization thread
 * @param sync Synchronization flags among threaded functions: producer, consumer, and visualizer.
 * @param dir Directory where the function dequeues and saves captured frames
 * @param verbose Prints enqueuing information
 * @return
 */
void consume_realsense(uls::SafeQueue<rs_frame_t> & q, 
                       frame_buffer_t & last_frame,
                       synchronization_t & sync, 
                       fs::path dir, 
                       boost::format fmt = boost::format("%08d"),
                       bool verbose = false)
{
    std::vector<int> compression_params = {cv::IMWRITE_PNG_COMPRESSION, 0};
    
    std::ofstream outfile;

    int fid = 0;
    while (sync.capture) // || q.size() > 0)
    {
        rs_frame_t frame = q.dequeue(); // the queue is thread safe ...
        // but he last_frame object it's not, unless we lock the object
        last_frame.lock();
        last_frame.rs = frame;
        last_frame.unlock();

        if (verbose)
            std::cout << "[RS] CONSUMER <<D " << q.size() << " ..." << '\n';

        if (sync.save_started && !sync.save_suspended && !dir.empty())
        {
            // long time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(last_frame.rs.ts.time_since_epoch()).count();
            // fmt % fid;
            // outfile.open((dir / "rs/color.log").string(), std::ios_base::app);
            // outfile << fmt.str() << ',' << std::to_string(time_ms) << '\n';
            // outfile.close();

            // fs::path color_dir = dir / "rs/color/";
            // fs::path depth_dir = dir / "rs/depth/";
            // if (!last_frame.rs.img_c.empty())
            //     cv::imwrite((color_dir / ("c_" + fmt.str() + ".jpg")).string(), last_frame.rs.img_c.clone());
            // if (!last_frame.rs.img_d.empty())
            //     cv::imwrite((depth_dir / ("d_" + fmt.str() + ".png")).string(), last_frame.rs.img_d.clone(), compression_params);
            std::string color_file = "c_" + fmt.str() + ".jpg";
            std::string depth_file = "d_" + fmt.str() + ".png";

            int64_t time_rs = std::chrono::duration_cast<std::chrono::milliseconds>(last_frame.rs.ts.time_since_epoch()).count();
            
            if (!last_frame.rs.img_c.empty())
            {
                outfile.open((dir / "rs/color.log").string(), std::ios_base::app);
                outfile << color_file << ',' << std::to_string(time_rs) << '\n';
                outfile.close();
                cv::imwrite((dir / "rs/color/" / color_file).string(), last_frame.rs.img_c);
            }
            if (!last_frame.rs.img_d.empty())
            {
                outfile.open((dir / "rs/depth.log").string(), std::ios_base::app);
                outfile << depth_file << ',' << std::to_string(time_rs) << '\n';
                outfile.close();
                cv::imwrite((dir / "rs/depth/" / depth_file).string(), last_frame.rs.img_d, compression_params);
            }
        }

        fid++;
    }
}

/*
 * Dequeues purethermal's frames from a thread-safe queue.
 * 
 * It keeps consuming until the synchronization object tells it to stop.
 * 
 * @param q Is the safe-thread queue of PureThermal's frames
 * @param last_frame Keep a buffer of the last consumed frame for the visualization thread
 * @param sync Synchronization flags among threaded functions: producer, consumer, and visualizer.
 * @param dir Directory where the function dequeues and saves captured frames
 * @param verbose Prints enqueuing information
 * @return
 */
void consume_purethermal(uls::SafeQueue<pt_frame_t> & q, 
                        //  pt_frame_t & capture,
                         frame_buffer_t & last_frame,
                        //  std::mutex & m,
                         synchronization_t & sync, 
                         fs::path dir, 
                        //  uls::MovementDetector md, 
                         boost::format fmt = boost::format("%08d"),
                         bool verbose = false)
{
    std::vector<int> compression_params = {cv::IMWRITE_PNG_COMPRESSION, 0};
    
    std::ofstream outfile;
    
    // auto start = std::chrono::steady_clock::now();
    // cv::Mat capture_img_prev;

    int fid = 0;
    while (sync.capture) // || q.size() > 0)
    {
        pt_frame_t frame = q.dequeue(); // q is thread safe
        // last_frame it's not, unless we invoke lock()
        last_frame.lock();
        last_frame.pt = frame;
        last_frame.unlock();

        if (verbose)
            std::cout << "[PT] CONSUMER <<D " << q.size() << " ..." << '\n';

        if (sync.save_started && !sync.save_suspended && !dir.empty())
        {
            std::string thermal_file = "t_" + fmt.str() + ".png";

            int64_t time_pt = std::chrono::duration_cast<std::chrono::milliseconds>(last_frame.pt.ts.time_since_epoch()).count();
            fmt % fid;
            outfile.open((dir / "pt/thermal.log").string(), std::ios_base::app);
            outfile << thermal_file << ',' << std::to_string(time_pt) << '\n';
            outfile.close();

            cv::imwrite((dir / "pt/thermal/" / thermal_file).string(), last_frame.pt.img, compression_params);
            // long time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(last_frame.pt.ts.time_since_epoch()).count();
            // fmt % fid;
            // outfile.open((dir / "pt/thermal.log").string(), std::ios_base::app);
            // outfile << fmt.str() << ',' << std::to_string(time_ms) << '\n';
            // outfile.close();
            
            // fs::path thermal_dir = dir / "pt/thermal/";
            // cv::imwrite((thermal_dir / ("t_" + fmt.str() + ".png")).string(), last_frame.pt.img, compression_params);
        }

        fid++;
    }
}

/*
 * Dequeues purethermal's frames from a thread-safe queue.
 * 
 * It keeps consuming until the synchronization object tells it to stop.
 * 
 * @param q Is the safe-thread queue
 * @param is_capturing The boolean flag indicating if the function must keep dequeueing captured realsense frames
 * @param dir Directory where the function dequeues and saves captured frames
 * @param verbose Prints enqueuing information
 * @return
 */
// void consume(uls::SafeQueue<rs_frame_t> & q_rs, 
//              uls::SafeQueue<pt_frame_t> & q_pt, 
//              frame_buffer_t & last_frame,
//              synchronization_t & sync, 
//              fs::path dir, 
//             //  uls::MovementDetector md, 
//              int duration,
//              int64_t delay,
//              bool verbose)
// {
//     std::vector<int> compression_params = {cv::IMWRITE_PNG_COMPRESSION, 0};
    
//     int fid = 0;
//     boost::format fmt("%08d");
//     std::ofstream outfile;
    
//     auto start = std::chrono::steady_clock::now();
//     cv::Mat capture_img_prev;

//     std::vector<rs_frame_t> buffer;
//     while (sync.capture) try
//     {
//         pt_frame_t capture_pt = q_pt.dequeue();
//         if (verbose)
//             std::cout << "[RS+PT] CONSUMER <<D(pt) " << q_pt.size() << "+" << q_rs.size() << '\n';

//         rs_frame_t closest_rs;

//         // Process buffer
//         int64_t min_diff = std::numeric_limits<int64_t>::max();
//         for (int k = 0; k < buffer.size(); k++)
//         {
//             int64_t diff = std::chrono::duration_cast<std::chrono::milliseconds>(capture_pt.ts - buffer[k].ts).count();
//             if (abs(diff) < min_diff)
//             {
//                 min_diff = abs(diff);
//                 closest_rs = buffer[k];
//             }
//         }
//         buffer.clear();

//         // Process newly captured frames
//         rs_frame_t capture_rs = q_rs.dequeue();
//         if (verbose)
//             std::cout << "[RS+PT] CONSUMER <<D(rs) " << q_pt.size() << "+" << q_rs.size() << '\n';

//         int64_t diff = std::chrono::duration_cast<std::chrono::milliseconds>(capture_pt.ts - capture_rs.ts).count();        
//         while (diff > -delay)
//         {
//             if (diff < delay)
//             {
//                 if (abs(diff) < min_diff)
//                 {
//                     min_diff = abs(diff);
//                     closest_rs = capture_rs;
//                     buffer.clear();
//                     buffer.resize(1, closest_rs);
//                 }
//                 else
//                 {
//                     buffer.push_back(capture_rs);
//                 }
//             }

//             capture_rs = q_rs.dequeue();
//             if (verbose)
//                 std::cout << "[RS+PT] CONSUMER <<D(rs) " << q_pt.size() << "+" << q_rs.size() << '\n';

//             diff = std::chrono::duration_cast<std::chrono::milliseconds>(capture_pt.ts - capture_rs.ts).count();
//         }

//         // last frame is a struct that derives from mutex
//         // last_frame.lock(); // make sure to lock before modifying the pair of (rs,pt) frames
//         last_frame.pt = capture_pt;
//         last_frame.rs = closest_rs;
//         // last_frame.unlock();

//         if (sync.save_started && !sync.save_suspended && !dir.empty())
//         {
//             int64_t time_pt = std::chrono::duration_cast<std::chrono::milliseconds>(capture_pt.ts.time_since_epoch()).count();
//             fmt % fid;
//             outfile.open((dir / "pt.log").string(), std::ios_base::app);
//             outfile << fmt.str() << ',' << std::to_string(time_pt) << '\n';
//             outfile.close();
//             int64_t time_rs = std::chrono::duration_cast<std::chrono::milliseconds>(closest_rs.ts.time_since_epoch()).count();
//             outfile.open((dir / "rs.log").string(), std::ios_base::app);
//             outfile << fmt.str() << ',' << std::to_string(time_rs) << '\n';
//             outfile.close();
            
//             std::cout << "Try to save " << capture_pt.img.size() << '%' << capture_rs.img_c.size() << '%' << capture_rs.img_d.size() << std::endl;
//             // cv::imwrite((dir / "pt/thermal/" / ("t_" + fmt.str() + ".png")).string(), capture_pt.img, compression_params);
//             // cv::imwrite((dir / "rs/color/" / ("c_" + fmt.str() + ".jpg")).string(), closest_rs.img_c);
//             // cv::imwrite((dir / "rs/depth/" / ("d_" + fmt.str() + ".png")).string(), closest_rs.img_d, compression_params);
//             std::cout << "Finished try" << std::endl;
//         }
            
//         fid++;
//     }         
//     catch (uls::dequeue_error & e)
//     {
//         std::cout << e.what() << std::endl;
//     }
// }

/*
 * Dequeues multimodal frames from a thread-safe queue.
 * 
 * It keeps consuming until the synchronization object tells it to stop.
 * 
 * @param q Is the safe-thread queue of PureThermal's frames
 * @param last_frame Keep a buffer of the last consumed frame for the visualization thread
 * @param sync Synchronization flags among threaded functions: producer, consumer, and visualizer.
 * @param dir Directory where the function dequeues and saves captured frames
 * @param verbose Prints enqueuing information
 * @return
 */
void consume(uls::SafeQueue<std::pair<rs_frame_t,pt_frame_t> > & q,
             frame_buffer_t & last_frame,
             synchronization_t & sync, 
             fs::path dir, 
            //  uls::MovementDetector md, 
            //  int duration,
             int64_t delay,
             boost::format fmt = boost::format("%08d"),
             bool verbose = false)
{
    std::vector<int> compression_params = {cv::IMWRITE_PNG_COMPRESSION, 0};
    
    std::ofstream outfile;
    
    auto start = std::chrono::steady_clock::now();
    cv::Mat capture_img_prev;

    // std::vector<rs_frame_t> buffer;
    int fid = 0;
    while (sync.capture) try
    {
        std::pair<rs_frame_t,pt_frame_t> capture = q.dequeue();

        // TODO: lock? Does it cause deadlock with visualization thread?
        last_frame.rs = capture.first;
        last_frame.pt = capture.second;
        
        if (sync.save_started && !sync.save_suspended && !dir.empty())
        {
            int64_t time_rs = std::chrono::duration_cast<std::chrono::milliseconds>(last_frame.rs.ts.time_since_epoch()).count();

            if (!last_frame.rs.img_c.empty())
            {
                std::string color_file = "c_" + fmt.str() + ".jpg";

                outfile.open((dir / "rs/color.log").string(), std::ios_base::app);
                outfile << color_file << ',' << std::to_string(time_rs) << '\n';
                outfile.close();

                cv::imwrite((dir / "rs/color/" / color_file).string(), last_frame.rs.img_c);
            }

            if (!last_frame.rs.img_d.empty())
            {
                std::string depth_file = "d_" + fmt.str() + ".png";

                outfile.open((dir / "rs/depth.log").string(), std::ios_base::app);
                outfile << depth_file << ',' << std::to_string(time_rs) << '\n';
                outfile.close();

                cv::imwrite((dir / "rs/depth/" / depth_file).string(), last_frame.rs.img_d, compression_params);
            }

            int64_t time_pt = std::chrono::duration_cast<std::chrono::milliseconds>(last_frame.pt.ts.time_since_epoch()).count();

            std::string thermal_file = "t_" + fmt.str() + ".png";

            outfile.open((dir / "pt/thermal.log").string(), std::ios_base::app);
            outfile << thermal_file << ',' << std::to_string(time_pt) << '\n';
            outfile.close();

            cv::imwrite((dir / "pt/thermal/" / thermal_file).string(), last_frame.pt.img, compression_params);
        }
            
        fid++;
    }         
    catch (uls::dequeue_error & e)
    {
        std::cout << e.what() << std::endl;
    }
}

/*
 * Timer function switching is_capturing from true to false after some time.
 * 
 * @param duration Number of millisecons after which is_capturing is put to false.
 * @param b A boolean will be switched, either 0->1 or 1->0, after duration.
 * @param elapsed Keeps count of elapsed time.
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

/*
 * Performs chessboard detection on an image given a pattern_size and draws it on top of the image.
 * 
 * @param img Image where to find the chessboard corners
 * @param pattern_size Size of the chessboard pattern
 * @flags Pattern detection flags (see cv::findChessboardCorners's "flags" parameter values)
 * @return
 */
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

/*
 * Detection function to be run in a thread.
 * 
 * @param md The detector itself.
 * @param sync Synchronization flags to be modified according to the output of the detector.
 * @param frame Current frame to evaluate in the detector.
 * @param duration Keep the function going until duration is expired.
 * @return
 */
// void detection(uls::MovementDetector & md, 
//                synchronization_t & sync,
//                frame_buffer_t & frame, 
//                std::chrono::milliseconds duration)
// {
//     auto start = std::chrono::high_resolution_clock::now();
//     while (sync.capture)
//     {
//         if (sync.save_started)
//         {
//             frame.lock();
//             if ( !frame.pt.img.empty() && md.find(frame.pt.img) )
//                 start = std::chrono::high_resolution_clock::now();
//             frame.unlock();
            
//             std::chrono::duration<double,std::milli> elapsed (std::chrono::high_resolution_clock::now() - start);
//             sync.save_suspended = (duration.count() > 0 && elapsed > duration);
//         }
//     }
// }
template <typename T>
void detection(T & detector, 
               synchronization_t & sync,
               frame_buffer_t & frame, 
               std::chrono::milliseconds duration)
{
    auto start = std::chrono::high_resolution_clock::now();
    while (sync.capture)
    {
        if (sync.save_started)
        {
            frame.lock();
            if ( !frame.pt.img.empty() && detector.find(frame.pt.img) )
                start = std::chrono::high_resolution_clock::now();
            frame.unlock();
            
            std::chrono::duration<double,std::milli> elapsed (std::chrono::high_resolution_clock::now() - start);
            sync.save_suspended = (duration.count() > 0 && elapsed > duration);
        }
    }
}

/*
 * Visualizer :)
 * 
 * @param win_name Window name where to visualize the consumed frames.
 * @param frame Last consumed frame.
 * @param sync Synchronization flags.
 * @param depth_scale Factor to transform depth reading to actual measurements.
 * @param intrinsics Intrinsic parameters of the different modalities to be visualized
 * @param extrinsics Extrinsic parameters (rotation and translation) to calibrate, e.g. thermal -> depth
 * @param find_pattern_flags Indicates whether or not to find chessboard in each modality.
 * @param pattern_size Size of the patter to be found if indicated via find_pattern_flags. 
 * @param hide_rgb Hide the RGB modality during visualization.
 */
void visualizer(std::string win_name,
               frame_buffer_t & frame,
            //    rs_frame_t & rs_frame, 
            //    pt_frame_t & pt_frame, 
            //    std::mutex & m,
            //    int modality_flags,
               synchronization_t & sync, 
               float depth_scale = 1.0,
               std::shared_ptr<std::map<std::string, uls::intrinsics_t> > intrinsics = NULL,
               std::shared_ptr<uls::extrinsics_t> extrinsics = NULL,
               int find_pattern_flags = 0,
               cv::Size pattern_size = cv::Size(6,5),
               bool hide_rgb = false
)
{
    // bool use_depth = (modality_flags & USE_DEPTH) > 0;
    std::vector<cv::Mat> frames (4);
    while (sync.capture)
    {
        cv::Mat depth;

        // m.lock();
        // cv::Mat rs_c = rs_frame.img_c;
        // cv::Mat rs_d = rs_frame.img_d;
        // cv::Mat pt = pt_frame.img;
        // m.unlock();
        frame.lock();
        cv::Mat rs_c = frame.rs.img_c;
        cv::Mat rs_d = frame.rs.img_d;
        cv::Mat pt = frame.pt.img;
        frame.unlock();

        if (!rs_c.empty() && !hide_rgb)
        {
            cv::Mat color;

            if (intrinsics)
            {
                color.create(rs_c.size(), rs_c.type());
                cv::undistort(rs_c, color, (*intrinsics)["rs/color,rs/depth"].camera_matrix, (*intrinsics)["rs/color,rs/depth"].dist_coeffs);
            }
            else
                color = rs_c.clone();

            if (find_pattern_flags & FIND_PATTERN_ON_COLOR) 
                find_and_draw_chessboard(color, pattern_size, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE/* + cv::CALIB_CB_FAST_CHECK + cv::CALIB_CB_FAST_CHECK*/);

            frames[0] = color;
        }

        if (!rs_d.empty())
        {
            if (intrinsics)
            {
                depth.create(rs_d.size(), rs_d.type());
                cv::undistort(rs_d, depth, (*intrinsics)["rs/color,rs/depth"].camera_matrix, (*intrinsics)["rs/color,rs/depth"].dist_coeffs);
            }
            else
                depth = rs_d.clone();

            depth.setTo(0, depth > 5000);
            uls::depth_to_8bit(depth, frames[1]);
        }


        // PureThermal-related preprocessing
        // pt_frame_t pt_frame = q_pt.peek_back();
        if (!pt.empty())
        {
            cv::Mat therm, therm_a;
            uls::thermal_to_8bit(pt.clone(), therm);
            uls::resize(therm, therm, cv::Size(1280,720));
            
            if (intrinsics)
            {
                cv::Mat tmp (therm.size(), therm.type());
                cv::undistort(therm, tmp, (*intrinsics)["pt/thermal"].camera_matrix, (*intrinsics)["pt/thermal"].dist_coeffs);
                therm = tmp;

                if (extrinsics && !depth.empty())
                {
                    cv::Mat map_x, map_y;
                    align_to_depth(depth, (*intrinsics)["rs/color,rs/depth"].camera_matrix, (*intrinsics)["pt/thermal"].camera_matrix, depth_scale, extrinsics, map_x, map_y);
                    cv::remap(therm, therm_a, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

                    frames[3] = therm_a;
                }
            }
            else if (find_pattern_flags & FIND_PATTERN_ON_THERM)
            {
                cv::cvtColor(therm, therm, cv::COLOR_GRAY2BGR);
                find_and_draw_chessboard(therm, pattern_size, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
            }
            
            frames[2] = therm;
        }

        // When hide_rgb is True, print project information instead.
        if (hide_rgb)
        {
            // frames[0] = cv::Mat::zeros(cv::Size(1280,720), CV_8UC3);
            std::vector<std::string> title_content = {
                "SENIOR project (S&C and CVC)"
            };

            std::vector<std::string> content = {
                "Capturing data of (anonymized) people passing by",
                "to then train a Depth+Thermal human detector."
            };
            std::vector<std::string> foot_content = {
                "[+] Contact:", 
                " |- aclapes@cvc.uab.es",
                " '- sescalera@cvc.uab.es"
            };

            uls::three_part_text(frames[0], title_content, content, foot_content, cv::Size(640, 320), 1.5, 0.75, 0.75, 1);
        }
        else if (!frames[0].empty()) // if not, show rgb
        {
            cv::putText(frames[0], "Color", 
                        cv::Point(30, 40), 
                        cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, cv::Scalar(0,0,255), 2, 8, false);
        }

        // the rest of the modalities are always visualized
        cv::putText(frames[1], "Depth", 
                    cv::Point(30,40), 
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, cv::Scalar(0,0,255), 2, 8, false);
        cv::putText(frames[2], "Thermal", 
                    cv::Point(30,40), 
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, cv::Scalar(0,0,255), 2, 8, false);
        cv::putText(frames[3], "Thermal (rescaled) -> Depth", 
                    cv::Point(30,40), 
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, cv::Scalar(0,0,255), 2, 8, false);

        // tile the images of the different modalites
        cv::Mat tiling;
        uls::tile(frames, 1280, 720, 2, 2, tiling); // in a 2x2 grid
        if (sync.save_started && !sync.save_suspended) // when frames are consumed, show red rectangle around the tiling
            cv::rectangle(tiling, cv::Rect(0, 0, tiling.cols, tiling.rows), cv::Scalar(0,0,255), 10);

        // Show up-and-running time in the viewer
        std::stringstream ss;
        ss << "Up-time (sec): " << std::chrono::duration_cast<std::chrono::seconds>(sync.capture_elapsed - sync.save_elapsed).count();
        int baseline;
        cv::Size uptime_text_size = cv::getTextSize(ss.str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, 1, &baseline);
        cv::rectangle(tiling, 
                      cv::Point(0, tiling.rows - (uptime_text_size.height + baseline)), 
                      cv::Point(uptime_text_size.width, tiling.rows), cv::Scalar(0,0,0), -1);
        cv::putText(tiling, ss.str(), 
                    cv::Point(0,tiling.rows - baseline), 
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, cv::Scalar(255,255,255), 1, 8, false);

        cv::imshow(win_name, tiling);
        cv::waitKey(1);
    }
}


int main(int argc, char * argv[]) try
{    
    /* -------------------------------- */
    /*   Command line argument parsing  */
    /* -------------------------------- */

    bool hide_rgb = false;

    po::options_description desc("Program options");
    desc.add_options()
        ("help,h", "Print help messages")
        ("duration,d", po::value<int>()->default_value(8000), "Duration of the recording in milliseconds (ms)")
        ("modalities,M", po::value<std::string>()->default_value("color,depth,thermal"), "Comma-separated list of modalities to capture")
        ("find-pattern-on,F", po::value<std::string>()->implicit_value(""), "Comma-separated list of modalities to find pattern on (color and/or thermal)")
        ("pattern,p", po::value<std::string>()->default_value("8,7"), "Pattern size \"x,y\" squares")
        ("fps,f", po::value<int>()->default_value(30), "Acquisition speed (fps) of realsense (integer number 1~30)")
        ("calibration-params", po::value<std::string>()->default_value(""), "Calibration mapping parameters")
        ("sync-delay", po::value<int>()->default_value(30), "Maximum time delay between RS and PT (in milliseconds)")
        ("md-duration", po::value<int>(), "When movement detected record during X millisecs")
        ("md-pixel-thresh", po::value<int>()->default_value(15), "When movement detected record during X millisecs")
        ("md-frame-ratio", po::value<float>()->default_value(0.01), "When movement detected record during X millisecs")
        ("hide-rgb,", po::bool_switch(&hide_rgb), "Hide RGB modality")
        ("verbosity,v", po::value<int>()->default_value(0), "Verbosity level (0: nothing | 1: countdown & output | 2: sections | 3: threads | 4: rs internals)")
        ("filename-fmt,", po::value<std::string>()->default_value("%08d"), "Output filename incremental identifier format.")
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

    if (verbosity > 4) 
        rs2::log_to_console(RS2_LOG_SEVERITY_DEBUG);

    /* Define input modalities */

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

    /* Define chessboard search in Color/Thermal modalities */

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

    /* Define calibration pattern size */

    std::vector<std::string> pattern_dims;
    boost::split(pattern_dims, vm["pattern"].as<std::string>(), boost::is_any_of(","));
    assert(pattern_dims.size() == 2);
    
    int x = std::stoi(pattern_dims[0]);
    int y = std::stoi(pattern_dims[1]);
    assert(x > 2 && x > 2);

    cv::Size pattern_size (x,y);

    std::shared_ptr<std::map<std::string,uls::intrinsics_t> > intrinsics;
    std::shared_ptr<uls::extrinsics_t> extrinsics;
    if (!vm["calibration-params"].as<std::string>().empty())
    {
        cv::FileStorage fs (vm["calibration-params"].as<std::string>(), cv::FileStorage::READ);
        if (fs.isOpened())
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
        }
    }

    /* Initialize devices */
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
    
    float depth_scale = 1e-3;
    if (modality_flags & (USE_COLOR | USE_DEPTH))
    {
        profile = pipe_rs.start(cfg_rs);
        if (modality_flags & USE_DEPTH) depth_scale = rs_help::get_depth_scale(profile.get_device());
    }

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

    /* Create output directory structures to store captured data */

    std::string date_and_time = uls::current_time_and_date();

    fs::path parent (vm["output-dir"].as<std::string>());
    if (!parent.empty())
    {
        if (verbosity > 1) 
            std::cout << "[Main] Creating output directory structure ...\n";

        parent = parent / fs::path(date_and_time);
        if (modality_flags & (USE_DEPTH | USE_COLOR) )
        {
            if (modality_flags & USE_COLOR) 
                boost::filesystem::create_directories(parent / fs::path("rs/color/"));
            if (modality_flags & USE_DEPTH) 
                boost::filesystem::create_directories(parent / fs::path("rs/depth/"));
            
            cv::FileStorage fs ((parent / fs::path("rs_info.yml")).string(), cv::FileStorage::WRITE);
            if (fs.isOpened())
            {
                fs << "rs-serial_number" << profile.get_device().get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
                fs << "rs-depth_scale" << depth_scale;
                // ...
                fs.release();
            }
        }
        if (modality_flags & USE_THERM) 
        {
            boost::filesystem::create_directories(parent / fs::path("pt/thermal/"));
            // cv::FileStorage fs ((parent / fs::path("devices_info.yml")).string(), cv::FileStorage::APPEND);
            // if (fs.isOpened())
            // {
            //     fs << "pt-serial_number" << ""; // to keep generality (serial number is not provided by PT)
            //     // ...
            //     fs.release();
            // }
        }

        if (extrinsics)
        {
            fs::path calibration_params_path (vm["calibration-params"].as<std::string>());
            fs::copy_file(calibration_params_path, (parent / fs::path("calibration.yml")).string());
            cv::FileStorage fs ((parent / fs::path("calibration.yml")).string(), cv::FileStorage::APPEND);
            if (fs.isOpened())
                fs << "original_file" << fs::basename(calibration_params_path);
                fs.release();
        }

        if (verbosity > 1) 
            std::cout << "[Main] Output directory structure \"" << parent.string() << "\"created\n";
    }

    /*
     * Initialize consumer-producer queues
     */

    std::thread p_rs_thr, p_pt_thr, p_thr; // producers
    std::thread c_thr; // consumer

    synchronization_t sync; // thread synchronization flags

    uls::SafeQueue<rs_frame_t> prod_to_cons_rs; // producer to consumer frame queues
    uls::SafeQueue<pt_frame_t> prod_to_cons_pt;
    uls::SafeQueue<std::pair<rs_frame_t, pt_frame_t> > prod_to_cons;

    // Producer threads initialization

    if (verbosity > 1) std::cout << "[Main] Starting RS consumer/producer threads ...\n";

    // rs/pt-related producers
    if ((modality_flags & (USE_COLOR | USE_DEPTH)) > 0 && (modality_flags & USE_THERM) > 0)
    {
        p_thr = std::thread(produce,
                            std::ref(pipe_rs),
                            std::ref(pipe_pt),
                            std::ref(profile),
                            std::ref(prod_to_cons),
                            std::ref(sync),
                            modality_flags,
                            verbosity > 2
        );
    }
    else if (modality_flags & (USE_COLOR | USE_DEPTH))
    {
        p_rs_thr = std::thread(produce_realsense, 
                               std::ref(pipe_rs), 
                               std::ref(profile), 
                               std::ref(prod_to_cons_rs), 
                               std::ref(sync), 
                               modality_flags, 
                               verbosity > 2);
    }
    else if (modality_flags & USE_THERM)
    {
        p_pt_thr = std::thread(produce_purethermal, 
                               std::ref(pipe_pt), 
                               std::ref(prod_to_cons_pt), 
                               std::ref(sync), 
                               verbosity > 2);
    }

    if (verbosity > 1) std::cout << "[Main] Producer threads started ...\n";

    // Consumer threads initialization
    frame_buffer_t last_frame; // one-frame buffer from consumer to viewer

    if (verbosity > 1) std::cout << "[Main] Starting consumer threads ...\n";

    // A movement detector can govern the sync.suspended flag to suspend the consumption and/or visualization
    // when no movement is detected
    std::thread md_thr;
    uls::MovementDetector md (vm["md-pixel-thresh"].as<int>(), vm["md-frame-ratio"].as<float>());
    md_thr = std::thread(detection<uls::MovementDetector>,
                         std::ref(md),
                         std::ref(sync),
                         std::ref(last_frame),
                         std::chrono::milliseconds(vm["md-duration"].as<int>()));

    // Create a consumer thread to process  produced frames (both rs and pt or only rs/pt)
    if ((modality_flags & (USE_COLOR | USE_DEPTH)) > 0 && (modality_flags & USE_THERM) > 0)
    {
        // c_thr = std::thread(consume, 
        //                     std::ref(prod_to_cons_rs), 
        //                     std::ref(prod_to_cons_pt), 
        //                     std::ref(last_frame),
        //                     std::ref(sync), 
        //                     parent, 
        //                     vm["md-duration"].as<int>(), 
        //                     vm["sync-delay"].as<int>(), 
        //                     verbosity > 2);
        c_thr = std::thread(consume, 
                    std::ref(prod_to_cons), 
                    std::ref(last_frame),
                    std::ref(sync), 
                    parent, 
                    // vm["md-duration"].as<int>(), 
                    vm["sync-delay"].as<int>(),
                    boost::format(vm["filename-fmt"].as<std::string>()),
                    verbosity > 2);
    }
    else if (modality_flags & (USE_COLOR | USE_DEPTH))
    {
        c_thr = std::thread(consume_realsense,
                            std::ref(prod_to_cons_rs), 
                            std::ref(last_frame), 
                            std::ref(sync),
                            parent, 
                            // vm["md-duration"].as<int>(), 
                            boost::format(vm["filename-fmt"].as<std::string>()),
                            verbosity > 2);
    }
    else
    {
        c_thr = std::thread(consume_purethermal, 
                            std::ref(prod_to_cons_pt), 
                            std::ref(last_frame), 
                            std::ref(sync), 
                            parent,
                            // vm["md-duration"].as<int>(), 
                            boost::format(vm["filename-fmt"].as<std::string>()),
                            verbosity > 2);
    }

    if (verbosity > 1) 
    {
        std::cout << "[Main] Consumer threads started ...\n";
        std::cout << "[Main] Starting visualization ...\n";
    }

    /* 
     * Visualization loop
     * ------------------
     */

    cv::namedWindow("Viewer");

    std::chrono::duration<double,std::milli> capture_elapsed, saving_elapsed;
    std::thread save_start_timer(timer, 3000, std::ref(sync.save_started), std::ref(sync.save_elapsed));
    std::thread capture_timer(timer, 3000 + vm["duration"].as<int>(), std::ref(sync.capture), std::ref(sync.capture_elapsed)); // Timer will set is_capturing=false when finished

    visualizer("Viewer", 
              last_frame,
              sync,
              depth_scale,
              intrinsics,
              extrinsics,
              find_pattern_flags,
              pattern_size,
              hide_rgb);

    cv::destroyWindow("Viewer");

    if (verbosity > 1) 
        std::cout << "[Main] Visualization (and capturing) ended ...\n";

    /* Wait for consumer threads to finish */

    if (verbosity > 1) 
        std::cout << "[Main] Joining all threads ...\n";

    save_start_timer.join();
    capture_timer.join();
    c_thr.join();
    md_thr.join();
    p_rs_thr.join();
    p_pt_thr.join();

    if (verbosity > 1) 
        std::cout << "[Main] All threads joined ...\n";

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

