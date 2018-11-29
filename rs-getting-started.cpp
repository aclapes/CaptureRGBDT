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

#include "pt_pipeline.hpp"
#include "safe_queue.hpp"

using timestamp_t = std::chrono::time_point<std::chrono::system_clock>;

bool debug = true;

void upfront_cv_window_hack()
{
    cv::namedWindow("GetFocus", CV_WINDOW_NORMAL);
    cv::Mat img = cv::Mat::zeros(1, 1, CV_8UC3);
    cv::imshow("GetFocus", img);
    cv::setWindowProperty("GetFocus", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    cv::waitKey(1);
    cv::setWindowProperty("GetFocus", CV_WND_PROP_FULLSCREEN, CV_WINDOW_NORMAL);
    cv::destroyWindow("GetFocus");
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

void produce_realsense(rs2::pipeline & p, SafeQueue<std::pair<std::pair<cv::Mat,cv::Mat>,timestamp_t>> & q, bool & is_capturing)
{
    rs2::colorizer color_map;
    while (is_capturing)
    {
        // Block program until frames arrive
        rs2::frameset frames = p.wait_for_frames();
        timestamp_t ts = std::chrono::system_clock::now();

        // Try to get a frame of a depth image
        rs2::depth_frame depth = frames.get_depth_frame();
        rs2::video_frame color = frames.get_color_frame();

        // Get the depth frame's dimensions
        float width = depth.get_width();
        float height = depth.get_height();

        // Query the distance from the camera to the object in the center of the image
//        float dist_to_center = depth.get_distance(width / 2, height / 2);

        // Print the distance
//        std::cout << "The camera is facing an object " << dist_to_center << " meters away         \r";
        cv::Mat img_c_ = cv::Mat(cv::Size(1280,720), CV_8UC3, const_cast<void *>(color.get_data()), cv::Mat::AUTO_STEP);
        cv::Mat img_d_ = cv::Mat(cv::Size(1280,720), CV_16U, const_cast<void *>(depth.get_data()), cv::Mat::AUTO_STEP);
//        rs2::frame depth_col = color_map.process(depth);
//        cv::Mat img_d_(cv::Size(1280, 720), CV_8UC3, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat img_c = img_c_.clone();
        cv::Mat img_d = img_d_.clone();
    q.enqueue(std::pair<std::pair<cv::Mat,cv::Mat>,timestamp_t>(std::pair<cv::Mat,cv::Mat>(img_c,img_d),ts));
        std::cerr << "[RS] <PRODUCER> enqueued " << q.size() << " ..." << '\n';
    }
}

void produce_purethermal(pt::pipeline & p, SafeQueue<std::pair<cv::Mat,timestamp_t>> & q, bool & is_capturing)
{
    while (is_capturing)
    {
        // Block program until frames arrive
        uvc_frame_t *frame = p.wait_for_frames();
        timestamp_t ts = std::chrono::system_clock::now();
        
        cv::Mat img_t_ = cv::Mat(cv::Size(160, 120), CV_16UC1, frame->data);
        cv::Mat img_t;
        cv::flip(img_t_, img_t, -1);
//        cv::Mat img_t = thermal_to_8bit(thermal_t);
//        cv::Mat img_gray, img_rgb;
//        img_t.convertTo(img_gray, CV_8UC3);
//        cvtColor(img_gray,img_rgb,CV_GRAY2RGB);
        q.enqueue(std::pair<cv::Mat,timestamp_t>(img_t,ts));
        std::cerr << "[PT] <PRODUCER> enqueued " << q.size() << " ..." << '\n';
        
        if (debug)
        {
            cv::Mat imgv_t = thermal_to_8bit(img_t);
            //            cv::Mat img_gray, img_rgb;
            //            img_t.convertTo(img_gray, CV_8UC3);
            //            cvtColor(img_gray,img_rgb,CV_GRAY2RGB);
            cv::flip(imgv_t, imgv_t, 1);
            //            cv::resize(img_t, img_t, cv::Size(640,480));
            cv::imshow("thermal dbg", imgv_t);
            cv::waitKey(1);
            //            cv::imwrite(dir + "/pt/thermal-dbg/t_" + fmt.str() + ".png", img_t, compression_params);
        }
    }
}

void consume_realsense(SafeQueue<std::pair<std::pair<cv::Mat,cv::Mat>,timestamp_t>> & q, bool & is_capturing, std::string dir)
{
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);
    
    int fid = 0;
    boost::format fmt("%08d");
    std::ofstream outfile(dir + "/rs.log", std::ios_base::app);;
    while (is_capturing || q.size() > 0)
    {
        std::pair<std::pair<cv::Mat,cv::Mat>,timestamp_t> capture = q.dequeue();
        std::cout << "[RS] >CONSUMER< dequeued " << q.size() << " ..." << '\n';

        long time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(capture.second.time_since_epoch()).count();
        fmt % fid;
        outfile << fmt.str() << ',' << std::to_string(time_ms) << '\n';
        
        cv::imwrite(dir + "/rs/color/c_" + fmt.str() + ".jpg", capture.first.first);
        cv::imwrite(dir + "/rs/depth/d_" + fmt.str() + ".png", capture.first.second, compression_params);
        
        fid++;
    }
}

void consume_purethermal(SafeQueue<std::pair<cv::Mat,timestamp_t>> & q, bool & is_capturing, std::string dir)
{
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);
    
    int fid = 0;
    boost::format fmt("%08d");
    std::ofstream outfile(dir + "/pt.log", std::ios_base::app);
    while (is_capturing || q.size() > 0)
    {
        std::pair<cv::Mat,timestamp_t> capture = q.dequeue();
        std::cout << "[PT] >CONSUMER< dequeued " << q.size() << " ..." << '\n';

        long time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(capture.second.time_since_epoch()).count();
        fmt % fid;
        outfile << fmt.str() << ',' << std::to_string(time_ms) << '\n';
        
        cv::imwrite(dir + "/pt/thermal/t_" + fmt.str() + ".png", capture.first, compression_params);

        fid++;
    }
}

void timer(bool & is_capturing, double time)
{
    auto start = std::chrono::high_resolution_clock::now();
    while (is_capturing)
    {
        // Control total time
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end-start;
        is_capturing = elapsed.count() <= time;
    }
}

int main(int argc, char * argv[]) //try
{
//    rs2::log_to_console(RS2_LOG_SEVERITY_DEBUG);
    double capture_time_ms = 20000;

    SafeQueue<std::pair<std::pair<cv::Mat,cv::Mat>,timestamp_t>> queue_rs;
    SafeQueue<std::pair<cv::Mat,timestamp_t>> queue_pt;
    bool is_capturing;
    
    std::string date_and_time = current_time_and_date();
    boost::filesystem::create_directories(date_and_time + "/rs/color/");
    boost::filesystem::create_directories(date_and_time + "/rs/depth/");
    boost::filesystem::create_directories(date_and_time + "/pt/thermal/");
    if (debug) boost::filesystem::create_directories(date_and_time + "/pt/thermal-dbg/");

    // Create a Pipeline - this serves as a top-level API for streaming and processing frames
    rs2::pipeline pipe_rs;
    pt::pipeline pipe_pt;
    
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 15);
//    cfg.enable_stream(RS2_STREAM_INFRARED, 1280, 720, RS2_FORMAT_Y8, 15);
    cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 15);
    
    // Configure and start the pipeline
    pipe_rs.start(cfg);
    pipe_pt.start();
    
    std::this_thread::sleep_for(std::chrono::seconds(1));
    is_capturing = true;
    
    std::cout << "Started capturing" << '\n';
    
    if (debug) {
        upfront_cv_window_hack();
        cv::namedWindow("thermal dbg");
    }

    /*
     * Producer threads read from sensors and enqueue them in a thread-safe queue object.
     */

    
    /*
     * Consumer threads wait for enqueued frames until all produced frames are saved to disk.
     */
    
    std::thread c_rs_thr(consume_realsense, std::ref(queue_rs), std::ref(is_capturing), date_and_time);
    std::thread c_pt_thr(consume_purethermal, std::ref(queue_pt), std::ref(is_capturing), date_and_time);
    
    /* Timer modifies the is_capturing flag after X millisecons */
    
    std::thread timer_thr(timer, std::ref(is_capturing), capture_time_ms);
    timer_thr.detach();
    
    std::thread p_rs_thr(produce_realsense, std::ref(pipe_rs), std::ref(queue_rs), std::ref(is_capturing));
//    std::thread p_pt_thr(produce_purethermal, std::ref(pipe_pt), std::ref(queue_pt), std::ref(is_capturing));
    produce_purethermal(pipe_pt, queue_pt, is_capturing);
    
    p_rs_thr.join();
//    p_pt_thr.join();
    
    pipe_rs.stop();
    pipe_pt.stop();
    
    /* Wait for consumer threads to finish */
    c_rs_thr.join();
    c_pt_thr.join();
    
    cv::destroyAllWindows();
    std::cout << "Sequence saved in " << date_and_time << '\n';

    return EXIT_SUCCESS;
}
//catch (const rs2::error & e)
//{
//    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
//    return EXIT_FAILURE;
//}
//catch (const std::exception & e)
//{
//    std::cerr << e.what() << std::endl;
//    return EXIT_FAILURE;
//}

