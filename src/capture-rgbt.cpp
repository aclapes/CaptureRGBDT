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

#include "utils/pt_pipeline.hpp"
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

typedef struct
{
    std::atomic<bool> capture {true};
    std::atomic<bool> save_started {false};
    std::atomic<bool> save_suspended {true};
    std::chrono::duration<double,std::milli> capture_elapsed;
    std::chrono::duration<double,std::milli> save_elapsed;
} synchronization_t;

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


bool visualize_rgb = true;

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
void produce_realsense(rs2::pipeline & pipe, rs2::pipeline_profile & profile, SafeQueue<rs_frame_t> & q, synchronization_t & sync, int modality_flags, bool verbose)
{
    rs2_stream align_to;
    // float depth_scale;
    rs2::align* p_align = NULL;

    if (modality_flags & USE_DEPTH)
    {
        align_to = find_stream_to_align(profile.get_streams());
        p_align = new rs2::align(align_to);
        // depth_scale = get_depth_scale(profile.get_device());
    }
    
    /*
    rs2::colorizer color_map;
    color_map.set_option(RS2_OPTION_HISTOGRAM_EQUALIZATION_ENABLED, 1.f);
    color_map.set_option(RS2_OPTION_COLOR_SCHEME, 2.f); // White to Black
    */
    while (sync.capture)
    {
        rs2::frameset frames = pipe.wait_for_frames(); // blocking instruction
        
        rs_frame_t frame;
        frame.ts = std::chrono::system_clock::now();
        
        if (((modality_flags & USE_COLOR) | (modality_flags & USE_DEPTH)) == (USE_COLOR | USE_DEPTH)) 
        {
            //    if (profile_changed(pipe.get_active_profile().get_streams(), profile.get_streams()))
            //     {a
            //         // If the profile was changed, update the align object, and also get the new device's depth scale
            //         profile = pipe.get_active_profile();
            //         align_to = find_stream_to_align(profile.get_streams());
            //         if (p_align != NULL) 
            //             delete p_align;
            //         p_align = new rs2::align(align_to);
            //         // depth_scale = get_depth_scale(profile.get_device());
            //     }

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
            // img_d.convertTo(img_d, CV_32F);
            // frame.img_d = img_d * depth_scale;
            frame.img_d = img_d.clone();
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
            // img_d.convertTo(img_d, CV_32F);
            // frame.img_d = img_d * depth_scale;
            frame.img_d = img_d.clone();
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
void produce_purethermal(pt::pipeline & p, SafeQueue<pt_frame_t> & q, synchronization_t & sync, bool verbose)
{
    while (sync.capture)
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
                       synchronization_t & sync, 
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

    // if (!dir.empty())
        // outfile.open((dir / "rs.log").string(), std::ios_base::app);

    // auto start = std::chrono::steady_clock::now();
    // cv::Mat capture_img_prev;
    

    while (sync.capture || q.size() > 0)
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

        if (sync.save_started && !sync.save_suspended && !dir.empty())
        {
            long time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(capture.ts.time_since_epoch()).count();
            fmt % fid;
            outfile.open((dir / "rs.log").string(), std::ios_base::app);
            outfile << fmt.str() << ',' << std::to_string(time_ms) << '\n';
            outfile.close();

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
                         synchronization_t & sync, 
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
    
    // if (!dir.empty())
        // outfile.open((dir / "pt.log").string(), std::ios_base::app);
    
    auto start = std::chrono::steady_clock::now();
    cv::Mat capture_img_prev;

    while (sync.capture || q.size() > 0)
    {
        pt_frame_t capture = q.dequeue();
        if (verbose)
            std::cout << "[PT] >CONSUMER< dequeued " << q.size() << " ..." << '\n';

        if (sync.save_started)
        {
            if (duration <= 0)
            {
                sync.save_suspended = false;
            }
            else
            {
                std::chrono::duration<double,std::milli> elapsed (std::chrono::steady_clock::now() - start);
                if ( elapsed >= std::chrono::milliseconds(duration) )
                {   
                    sync.save_suspended = true;
                    if ( duration <= 0 || md.find(capture.img) ) // if movement detected
                    {
                        start = std::chrono::steady_clock::now(); // reset clock
                        sync.save_suspended = false;
                    }
                }
            }

            if ( !sync.save_suspended && !dir.empty())
            {
                long time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(capture.ts.time_since_epoch()).count();
                fmt % fid;
                outfile.open((dir / "pt.log").string(), std::ios_base::app);
                outfile << fmt.str() << ',' << std::to_string(time_ms) << '\n';
                outfile.close();
                
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
 * Peeks and visualizes frames from RS and PT safe-queues.
 * 
 * TODO: write extended description
 * 
 * @param win_name OpenCV's window name where to visualize peeked frames
 * @param queue_rs Safe-queue containing RS-like frames (color and/or depth)
 * @param queue_pt Safe-queue containing PT-like frames (thermal)
 * @param intrinsics Intrinsic camera parameters for the different modalities ("Color" and/or "Thermal" are map keys)
 * @param modality_flags Flags indicating established modalities (see defines: USE_COLOR, USE_DEPTH, and/or USE_THERM)
 * @param sync Contains boolean flags and timers to control program flow and temporally sync threads.
 * @param extrinsics Rotation and translation matrices to spatially align queue_rs and queue_pt frames
 * @param find_pattern_flags Indicates modalities where to perform chessboard detection (see defines: FIND_PATTERN_ON_COLOR or _THERM)
 * @param pattern_size Size of the pattern to detect in modalities specified by "find_pattern_flags" parameter
 * @return
 */
void visualize(std::string win_name,
               SafeQueue<rs_frame_t> & queue_rs, 
               SafeQueue<pt_frame_t> & queue_pt, 
               int modality_flags,
               synchronization_t & sync, 
               float depth_scale = 1.0,
               std::shared_ptr<std::map<std::string, uls::intrinsics_t> > intrinsics = NULL,
               std::shared_ptr<uls::extrinsics_t> extrinsics = NULL,
               int find_pattern_flags = 0,
               cv::Size pattern_size = cv::Size(6,5)
)
{
    bool use_depth = (modality_flags & USE_DEPTH) > 0;

    std::vector<cv::Mat> frames (visualize_rgb ? 3 : 2);

    while (sync.capture)
    {
        rs_frame_t rs_frame;
        pt_frame_t pt_frame;

        cv::Mat depth;
        try 
        {
            rs_frame = queue_rs.peek(33);
            if (!rs_frame.img_c.empty() && visualize_rgb)
            {
                cv::Mat color;

                if (intrinsics)
                {
                    color.create(rs_frame.img_c.size(), rs_frame.img_c.type());
                    cv::undistort(rs_frame.img_c, color, (*intrinsics)["Color"].camera_matrix, (*intrinsics)["Color"].dist_coeffs);
                }
                else
                    color = rs_frame.img_c.clone();

                if (find_pattern_flags & FIND_PATTERN_ON_COLOR) 
                    uls::find_and_draw_chessboard(color, pattern_size, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE/* + cv::CALIB_CB_FAST_CHECK + cv::CALIB_CB_FAST_CHECK*/);

                frames[2] = color;
            }

            if (!rs_frame.img_d.empty())
            {
                if (intrinsics)
                {
                    depth.create(rs_frame.img_d.size(), rs_frame.img_d.type());
                    cv::undistort(rs_frame.img_d, depth, (*intrinsics)["Color"].camera_matrix, (*intrinsics)["Color"].dist_coeffs);
                }
                else
                    depth = rs_frame.img_d.clone();


                frames[0] = uls::DepthFrame::to_8bit(depth, cv::COLORMAP_BONE);
            }
        }
        catch (const std::runtime_error & e) // catch peek's exception
        {
            std::cout << e.what() << '\n';
        }
        
        // if ( !use_depth || (extrinsics && use_depth && !depth.empty()) )
        // {
            try
            {
                pt_frame = queue_pt.peek(33);
                if (!pt_frame.img.empty())
                {
                    cv::Mat therm = uls::ThermalFrame::to_8bit(pt_frame.img.clone());
                    uls::resize(therm, therm, cv::Size(1280,720));

                    if (intrinsics)
                    {
                        cv::Mat tmp = therm.clone();
                        cv::undistort(tmp, therm, (*intrinsics)["Thermal"].camera_matrix, (*intrinsics)["Thermal"].dist_coeffs);
                    }

                    if (extrinsics && use_depth && !depth.empty())
                    {
                        cv::Mat map_x, map_y;
                        align_to_depth(depth, (*intrinsics)["Color"].camera_matrix, (*intrinsics)["Thermal"].camera_matrix, depth_scale, extrinsics, map_x, map_y);
                        cv::remap(therm, therm, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
                    }
                    
                    if (find_pattern_flags & FIND_PATTERN_ON_THERM)
                    {
                        cv::cvtColor(therm, therm, cv::COLOR_GRAY2BGR);
                        uls::find_and_draw_chessboard(therm, pattern_size, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
                    }

                    frames[1] = therm;
                }
            }
            catch (const std::runtime_error & e) // catch peek's exception
            {
                std::cerr << e.what() << '\n';
            }
        // }

        cv::Mat tiling;
        uls::tile(frames, 640, 720, 1, frames.size(), tiling);
    
        tiling = (sync.save_started && !sync.save_suspended) ? tiling : ~tiling;

        std::stringstream ss;
        ss << (sync.capture_elapsed - sync.save_elapsed).count(); 
        cv::putText(tiling, ss.str(), cv::Point(tiling.cols/20.0,tiling.rows/20.0), CV_FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,0,255), 1, 8, false);

        cv::imshow(win_name, tiling);
        cv::waitKey(1);
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
            fs["camera_matrix_1"] >> (*intrinsics)[modality_1].camera_matrix;
            fs["camera_matrix_2"] >> (*intrinsics)[modality_2].camera_matrix;
            fs["dist_coeffs_1"]   >> (*intrinsics)[modality_1].dist_coeffs;
            fs["dist_coeffs_2"]   >> (*intrinsics)[modality_2].dist_coeffs;

            extrinsics = std::make_shared<uls::extrinsics_t>();
            fs["R"] >> extrinsics->R;
            fs["T"] >> extrinsics->T;
            // calibrate = true;
        }
    }

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
    
    float depth_scale = 1.0;
    if (modality_flags & (USE_COLOR | USE_DEPTH))
    {
        profile = pipe_rs.start(cfg_rs);
        if (modality_flags & USE_DEPTH) depth_scale = get_depth_scale(profile.get_device());
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

    /*
     * Initialize consumer-producer queues
     */

    synchronization_t sync;

    std::thread p_rs_thr, p_pt_thr; // producers
    std::thread c_rs_thr, c_pt_thr; // consumers

    SafeQueue<rs_frame_t> queue_rs (1); // set minimum buffer size to 1 (to be able to peek frames in visualize function)
    SafeQueue<pt_frame_t> queue_pt (1);

    // Producer threads initialization

    if (verbosity > 1) std::cout << "[Main] Starting RS consumer/producer threads ...\n";
    // rs-related producers
    if (modality_flags & (USE_COLOR | USE_DEPTH))
        p_rs_thr = std::thread(produce_realsense, std::ref(pipe_rs), std::ref(profile), std::ref(queue_rs), std::ref(sync), modality_flags, verbosity > 2);
    if (modality_flags & USE_THERM)
        p_pt_thr = std::thread(produce_purethermal, std::ref(pipe_pt), std::ref(queue_pt), std::ref(sync), verbosity > 2);

    if (verbosity > 1) std::cout << "[Main] Producer threads started ...\n";

    // Consumer threads initialization

    uls::MovementDetector md (vm["md-pixel-thresh"].as<int>(), vm["md-frame-ratio"].as<float>());

    if (verbosity > 1) std::cout << "[Main] Starting consumer threads ...\n";
    // rs-related consumers
    if (modality_flags & (USE_COLOR | USE_DEPTH))
        c_rs_thr = std::thread(consume_realsense, std::ref(queue_rs), std::ref(sync), 
                               parent, md, vm["md-duration"].as<int>(), verbosity > 2);
    if (modality_flags & USE_THERM)
        c_pt_thr = std::thread(consume_purethermal, std::ref(queue_pt), std::ref(sync), 
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
    std::thread save_start_timer(timer, 3000, std::ref(sync.save_started), std::ref(sync.save_elapsed));
    std::thread capture_timer(timer, 3000 + vm["duration"].as<int>(), std::ref(sync.capture), std::ref(sync.capture_elapsed)); // Timer will set is_capturing=false when finished
    // visualize("Viewer", queue_rs, queue_pt, sync, capture_elapsed, pattern_size, find_pattern_flags, calibrate ? &map_rs : NULL, calibrate ? &map_pt : NULL);
    visualize("Viewer", queue_rs, queue_pt, modality_flags, sync, depth_scale, intrinsics, extrinsics, find_pattern_flags, pattern_size);
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

