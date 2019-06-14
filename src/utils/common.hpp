//
//  utils.hpp
//  RealsenseExamplesGettingStarted
//
//  Created by Albert Clapés on 27/11/2018.
//

#ifndef utils_common_h
#define utils_common_h

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>

namespace fs = boost::filesystem;

namespace uls
{
    /*
     * Draws title, content, and foot text on an image.
     * 
     * @param img The image to draw on
     * @param title_content Title content
     * @param content Main text content (placed in the middle part of the image)
     * @param foot_content Bottom-part content
     * @param s Size of img
     * @param title_font_scale Scale of title's text font
     * @param font_scale Scale of content text font
     * @param foot_font_scale Scale of the foot text font
     * @param font_thickness Thickness of the fonts
     * @param font_type Font face
     * @param text_color Color of the text (Default: white)
     * @param bg_color Color of the background (Default: black)
     * @param interlinear Distance between drawn text lines
     * @param pad_left Left text padding relative as a factor relative to img's width
     * @return
     */
    void three_part_text(cv::Mat & img,
                     std::vector<std::string> title_content,
                     std::vector<std::string> content,
                     std::vector<std::string> foot_content,
                     cv::Size s = cv::Size(1280,720),
                     double title_font_scale = 3,
                     double font_scale = 1.5,
                     double foot_font_scale = 1.5,
                     int font_thickness = 2,
                     int font_type = cv::FONT_HERSHEY_PLAIN,
                     cv::Scalar text_color = cv::Scalar(255,255,255),
                     cv::Scalar bg_color = cv::Scalar(0,0,0),
                     double interlinear = 2.0, 
                     double pad_left = 0.05)
    {    
        img.create(s, CV_8UC3);
        img.setTo(bg_color);

        int height = s.height;
        int width = s.width;
        int baseline;
        cv::Size title_text_size, text_size, foot_font_size;

        title_text_size = cv::getTextSize(title_content[0], cv::FONT_HERSHEY_COMPLEX_SMALL, title_font_scale, font_thickness, &baseline);
        int title_height = interlinear * title_text_size.height * title_content.size();

        for (int i = 0; i < title_content.size(); i++)
        {
            cv::putText(img, title_content[i], cv::Point(pad_left*width, (i+1)*(title_height/title_content.size())),
                cv::FONT_HERSHEY_COMPLEX_SMALL, title_font_scale, text_color, font_thickness);
        }

        text_size = cv::getTextSize(content[0], cv::FONT_HERSHEY_COMPLEX_SMALL, font_scale, font_thickness, &baseline);
        int content_height = interlinear * text_size.height * content.size();

        for (int i = 0; i < content.size(); i++)
        {
            cv::putText(img, content[i], cv::Point(pad_left*width, height/2 - content_height/2 + i*(content_height/content.size())),
                cv::FONT_HERSHEY_COMPLEX_SMALL, font_scale, text_color, font_thickness);
        }

        foot_font_size = cv::getTextSize(foot_content[0], cv::FONT_HERSHEY_COMPLEX_SMALL, foot_font_scale, font_thickness, &baseline);
        int foot_height = interlinear * foot_font_size.height * foot_content.size();

        for (int i = 0; i < foot_content.size(); i++)
        {
            cv::putText(img, foot_content[i], cv::Point(pad_left*width, height - foot_height + i*(foot_height/foot_content.size())),
                cv::FONT_HERSHEY_COMPLEX_SMALL, font_scale, text_color, font_thickness);
        }
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
    * Resize image with automatic padding
    * 
    * This is a wrapper of cv::resize with automatic padding to keep aspect ratio
    * @src Source image
    * @dst Destiny image
    * @s Destiny image size
    * @return
    */
    cv::Rect resize(cv::Mat src, cv::Mat & dst, cv::Size s)
    {
        cv::Rect roi;

        if (s.empty() || (src.rows == s.height && src.cols == s.width))
        {
            // src.copyTo(dst);
            dst = src;
            roi = cv::Rect(0,0,dst.cols,dst.rows);
        }
        else
        {
            float ratio_s = ((float) s.width) / s.height;
            float ratio_src = ((float) src.cols) / src.rows;

            cv::Size new_domain, offset;

            if (ratio_src == ratio_s)
            {
                cv::resize(src, dst, s);
            }
            else if (ratio_src < ratio_s)
            {
                new_domain = cv::Size(s.height * ratio_src, s.height);
                offset = cv::Size((s.width - new_domain.height * ratio_src)/2., 0);
            }
            else
            {
                new_domain = cv::Size(s.width, s.width / ratio_src);
                offset = cv::Size(0, (s.height - new_domain.width / ratio_src)/2.);
            }

            cv::Mat tmp;
            cv::resize(src, tmp, new_domain);
            cv::copyMakeBorder(tmp, dst, 
                                offset.height, offset.height,
                                offset.width,  offset.width, 
                                cv::BORDER_CONSTANT, cv::Scalar::all(0));

            roi = cv::Rect(offset.width, offset.height, new_domain.width, new_domain.height);
        }

        return roi;
    }

    template<typename T>
    void minMax(cv::Mat m, T* minVal, T* maxVal, T ignVal)
    {
        *minVal = std::numeric_limits<T>::max();
        *maxVal = std::numeric_limits<T>::min();

        for (int i = 0; i < m.rows; i++)
        {
            for (int j = 0; j < m.cols; j++)
            {
                T val = m.at<T>(i,j);
                if (val != ignVal)
                { 
                    if (val < *minVal)
                        *minVal = val;
                    else if (val > *maxVal)
                        *maxVal = val;
                }
            }
        }
    }

    void thermal_to_8bit(cv::Mat src, 
                         cv::Mat & dst,
                         cv::Rect roi = cv::Rect(),
                         int colorMap = -1)
    {
        assert(src.type() == CV_16UC1);
        
        dst.release();

        if (roi.empty())
            roi = cv::Rect(0, 0, src.cols, src.rows);

        double minVal, maxVal;
        cv::minMaxIdx(src(roi), &minVal, &maxVal);

        cv::Mat tmp = src(roi);
        tmp.convertTo(tmp, CV_32FC1);
        tmp = 255 * (tmp - minVal) / (maxVal - minVal);
        tmp.convertTo(tmp, CV_8UC1);

        dst.create(src.rows, src.cols, CV_8UC1);
        dst.setTo(0);
        tmp.convertTo(dst(roi), CV_8UC1);

        if (colorMap > 0)
            applyColorMap(dst, dst, colorMap);
    }

    cv::Mat thermal_to_8bit(cv::Mat src)
    {
        cv::Mat dst;
        thermal_to_8bit(src, dst, cv::Rect(), -1);
        return dst;
    }

    cv::Mat color_to_8bit(cv::Mat src)
    {
        cv::Mat dst;
        cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
        return dst;
    }

    void depth_to_8bit(cv::Mat src, cv::Mat & dst, int colorMap = cv::COLORMAP_BONE, bool invertMap = true)
    {
        dst.release();

        double min, max;
        // unsigned short min, max;
        cv::minMaxIdx(src, &min, &max);
        // minMax<unsigned short>(src, &min, &max, 0);
        
        cv::Mat adjMap;
        // expand your range to 0..255. Similar to histEq();
        src.convertTo(adjMap, CV_8UC1, 255 / (max-min), -min);
        // src.setTo(0, src == -min);

        // this is great. It converts your grayscale image into a tone-mapped one, 
        // much more pleasing for the eye
        // function is found in contrib module, so include contrib.hpp 
        // and link accordingly
        // if (invertMap)
        // {
        //     cv::Mat err_mask = (adjMap == 0);
        //     adjMap = 255 - adjMap;
        //     adjMap.setTo(0, err_mask);
        // }
        cv::equalizeHist(adjMap, adjMap);
        applyColorMap(adjMap, dst, colorMap);
    }
    
    

//     class Frame
//     {
//         public:
//             Frame(fs::path path, cv::Size s = cv::Size())
//             {
//                 this->img = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
//                 this->roi = Frame::resize(this->img, this->img, s);
//             }

//             int get_type()
//             {
//                 return this->img.type();
//             }

//             // cv::Mat to_mat(cv::Size s)
//             // {
//             //     cv::Mat dst;
//             //     Frame::resize(img, dst, s);

//             //     return dst;
//             // }

//             /*
//             * Resize image with automatic padding
//             * 
//             * This is a wrapper of cv::resize with automatic padding to keep aspect ratio
//             * @src Source image
//             * @dst Destiny image
//             * @s Destiny image size
//             * @return
//             */
//             static cv::Rect resize(cv::Mat src, cv::Mat & dst, cv::Size s)
//             {
//                 cv::Rect roi;

//                 if (s.empty() || (src.rows == s.height && src.cols == s.width))
//                 {
//                     // src.copyTo(dst);
//                     dst = src;
//                     roi = cv::Rect(0,0,dst.cols,dst.rows);
//                 }
//                 else
//                 {
//                     float ratio_s = ((float) s.width) / s.height;
//                     float ratio_src = ((float) src.cols) / src.rows;

//                     cv::Size new_domain, offset;

//                     if (ratio_src == ratio_s)
//                     {
//                         cv::resize(src, dst, s);
//                     }
//                     else if (ratio_src < ratio_s)
//                     {
//                         new_domain = cv::Size(s.height * ratio_src, s.height);
//                         offset = cv::Size((s.width - new_domain.height * ratio_src)/2., 0);
//                     }
//                     else
//                     {
//                         new_domain = cv::Size(s.width, s.width / ratio_src);
//                         offset = cv::Size(0, (s.height - new_domain.width / ratio_src)/2.);
//                     }

//                     cv::Mat tmp;
//                     cv::resize(src, tmp, new_domain);
//                     cv::copyMakeBorder(tmp, dst, 
//                                         offset.height, offset.height,
//                                         offset.width,  offset.width, 
//                                         cv::BORDER_CONSTANT, cv::Scalar::all(0));

//                     roi = cv::Rect(offset.width, offset.height, new_domain.width, new_domain.height);
//                 }

//                 return roi;
//             }

//             void resize(cv::Size s)
//             {
//                 this->roi = Frame::resize(this->img(this->roi), this->img, s);
//             }

//             cv::Mat get()
//             {
//                 return this->img;
//             }

//         protected:
//             /*
//              * Actual "img" content (it excludes padding pixels)
//              * @return ROI indicating valid "img" area
//              */
//             cv::Rect get_roi()
//             {
//                 return this->roi;
//             }

//             /*
//              * Get "img" after applying the ROI to eliminate padding
//              * @return "img"'s valid area
//              */
//             cv::Mat get_no_padding()
//             {
//                 return this->img(this->roi);
//             }

//         private:
//             fs::path path;
//             int64_t time;
//             cv::Mat img;
//             cv::Rect roi;
//     };

//     /*
//      * Wrapper class for thermal images loaded on cv::Mat structures
//      * 
//      * It provides a function for converting raw sensor measurements to visually interpretable 8-bit values.
//      */
//     class ThermalFrame : public Frame
//     {
//         public:
//             const static unsigned short PADDING_VALUE = 0;

//             ThermalFrame(fs::path path, cv::Size s = cv::Size()) 
//               : Frame(path, s)
//             {
//                 assert(this->get_type() == CV_16UC1);
//             }

//             // static void to_8bit(cv::Mat src, cv::Mat & dst)
//             // {
//             //     cv::Mat tmp;
//             //     cv::normalize(src, tmp, 0, 65535, cv::NORM_MINMAX, CV_16UC1);
//             //     tmp.convertTo(dst, CV_8UC1, 1/256.);
//             // }

//             // static void to_8bit(cv::Mat src, cv::Mat & dst, cv::Size s)
//             // {
//             //     cv::Mat tmp;
//             //     ThermalFrame::to_8bit(src, tmp);
//             //     Frame::resize(tmp, dst, s);
//             // }

//             // static void to_8bit(cv::Mat src, cv::Mat & dst)
//             // {
//             //     dst.release();

//             //     double minVal, maxVal;
//             //     int minIdx, maxIdx;
//             //     cv::minMaxIdx(src, &minVal, &maxVal, &minIdx, &maxIdx);

//             //     dst.create(src.rows, src.cols, CV_8UC1, cv::Scalar::all(0));
//             //     cv::Mat tmp;
//             //     src.convertTo(tmp, CV_32FC1);
//             //     tmp = 255 * (tmp - minVal) / (maxVal - minVal);
//             //     tmp.convertTo(dst, CV_8UC1);
//             // }

//             static void to_8bit(cv::Mat src, cv::Rect roi, cv::Mat & dst)
//             {
//                 assert(src.type() == CV_16UC1);

//                 dst.release();

//                 // double minVal, maxVal;
//                 // cv::minMaxIdx(src(roi), &minVal, &maxVal);
//                 unsigned short minVal, maxVal;
//                 uls::minMax<unsigned short>(src(roi), &minVal, &maxVal, ThermalFrame::PADDING_VALUE);

//                 cv::Mat tmp = src(roi);
//                 cv::Mat mask = (tmp != ThermalFrame::PADDING_VALUE);

//                 tmp.convertTo(tmp, CV_32FC1);
//                 tmp = 255 * (tmp - minVal) / (maxVal - minVal);
//                 tmp.convertTo(tmp, CV_8UC1);

//                 dst.create(src.rows, src.cols, CV_8UC1);
//                 dst.setTo((unsigned char) ThermalFrame::PADDING_VALUE);
//                 tmp.copyTo(dst(roi), mask);
//                 // tmp.convertTo(dst(roi), CV_8UC1);

//             }

//             static void to_8bit(cv::Mat src, cv::Mat & dst)
//             {
//                 cv::Rect roi (0, 0, src.cols, src.rows);
//                 ThermalFrame::to_8bit(src, roi, dst);
//             }

//             cv::Mat to_8bit()
//             {
//                 cv::Mat dst;
//                 ThermalFrame::to_8bit(this->get(), this->get_roi(), dst);
//                 return dst;
//                 // double minVal, maxVal;
//                 // int minIdx, maxIdx;
//                 // cv::minMaxIdx(this->get_no_padding(), &minVal, &maxVal, &minIdx, &maxIdx);

//                 // cv::Mat dst = cv::Mat::zeros(this->get().rows, this->get().cols, CV_8UC1);
//                 // cv::Mat tmp;
//                 // this->get_no_padding().convertTo(tmp, CV_32FC1);
//                 // tmp = 255 * (tmp - minVal) / (maxVal - minVal);
//                 // tmp.convertTo(dst(this->get_roi()), CV_8UC1);

//                 // return dst;
//             }
//     };


// // class ThermalFrame : public Frame
// //     {
// //         public:
// //             ThermalFrame(fs::path path, cv::Size s = cv::Size(), int y_shift = 0)
// //               : Frame(path, s, y_shift)
// //             {
// //                 img = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
// //                 // img = ThermalFrame::to_8bit(img);
// //                 ThermalFrame::to_8bit(img, img);
// //                 resize(img, img, s);
// //                 if (y_shift > 0)
// //                     cv::copyMakeBorder(img, img, 0, y_shift, 0, 0, cv::BORDER_CONSTANT);
// //                 else if (y_shift < 0)
// //                     cv::copyMakeBorder(img, img, abs(y_shift), 0, 0, 0, cv::BORDER_CONSTANT);
// //             }

// //             // static cv::Mat to_8bit(cv::Mat data)
// //             // {
// //             //     cv::Mat img;
// //             //     double minVal, maxVal;
// //             //     cv::Point minIdx, maxIdx;
                
// //             //     cv::normalize(data, img, 0, 65535, cv::NORM_MINMAX);
// //             //     img.convertTo(img, CV_8UC1, 1/256.);
                
// //             //     return img;
// //             // }

// //             static void to_8bit(cv::Mat src, cv::Mat & dst)
// //             {
// //                 double minVal, maxVal;
// //                 int minIdx, maxIdx;
// //                 // cv::minMaxIdx(src, &minVal, &maxVal, &minIdx, &maxIdx);
// //                 // std::cout << '1' << minVal << ',' << maxIdx << std::endl;
// //                 cv::Mat tmp;
// //                 cv::normalize(src, tmp, 0, 65535, cv::NORM_MINMAX, CV_16UC1);
// //                 tmp.convertTo(dst, CV_8UC1, 1/256.);
// //                 // cv::minMaxIdx(dst,&minVal, &maxVal, &minIdx, &maxIdx);
// //                 // std::cout << '2' << minVal << ',' << maxIdx << std::endl;
// //             }

// //             void to_8bit(cv::Mat & dst)
// //             {
// //                 ThermalFrame::to_8bit(this->img, dst);
// //             }

// //             // cv::Mat mat() 
// //             // {
// //             //     return img;
// //             // }

// //         // private:
// //         //     cv::Mat img;
// //     };

//     // class ThermalFrame
//     // {
//     //     public:
//     //         ThermalFrame(fs::path path, cv::Size s = cv::Size(), int y_shift = 0)
//     //         {
//     //             img = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
//     //             // img = ThermalFrame::to_8bit(img);
//     //             std::cout << "Before resize: " << img.type() << std::endl;
//     //             resize(img, img, s);
//     //             std::cout << "after resize: " << img.type() << std::endl;
//     //             ThermalFrame::to_8bit(img, img);

//     //             if (y_shift > 0)
//     //                 cv::copyMakeBorder(img, img, 0, y_shift, 0, 0, cv::BORDER_CONSTANT);
//     //             else if (y_shift < 0)
//     //                 cv::copyMakeBorder(img, img, abs(y_shift), 0, 0, 0, cv::BORDER_CONSTANT);
//     //         }

//     //         // static cv::Mat to_8bit(cv::Mat data)
//     //         // {
//     //         //     cv::Mat img;
//     //         //     double minVal, maxVal;
//     //         //     cv::Point minIdx, maxIdx;
                
//     //         //     cv::normalize(data, img, 0, 65535, cv::NORM_MINMAX);
//     //         //     img.convertTo(img, CV_8UC1, 1/256.);
                
//     //         //     return img;
//     //         // }

//     //         static void to_8bit(cv::Mat src, cv::Mat & dst)
//     //         {
//     //             double minVal, maxVal;
//     //             int minIdx, maxIdx;
//     //             // cv::minMaxIdx(src, &minVal, &maxVal, &minIdx, &maxIdx);
//     //             // std::cout << '1' << minVal << ',' << maxIdx << std::endl;
//     //             cv::Mat tmp;
//     //             cv::normalize(src, tmp, 0, 65535, cv::NORM_MINMAX, CV_16UC1);
//     //             tmp.convertTo(dst, CV_8UC1, 1/256.);
//     //             // cv::minMaxIdx(dst,&minVal, &maxVal, &minIdx, &maxIdx);
//     //             // std::cout << '2' << minVal << ',' << maxIdx << std::endl;
//     //         }

//     //         void to_8bit(cv::Mat & dst)
//     //         {
//     //             ThermalFrame::to_8bit(this->img, dst);
//     //         }

//     //         cv::Mat mat() 
//     //         {
//     //             return img;
//     //         }

//     //     private:
//     //         cv::Mat img;
//     // };

//     /*
//      * Wrapper class for depth images loaded on cv::Mat structures
//      * 
//      * It provides a function for converting raw depth measurements to visually interpretable color-mapped 8-bit values.
//      */
//     class DepthFrame : public Frame
//     {
//         private:

//         public:
//             DepthFrame(fs::path path, cv::Size s = cv::Size()) 
//               : Frame(path, s)
//             {
//                 assert(this->get_type() == CV_16UC1);
//             }

//             /*
//             * Sam's code at https://stackoverflow.com/questions/13840013/opencv-how-to-visualize-a-depth-image
//             */
//             // static cv::Mat to_8bit(cv::Mat data, int colorMap = cv::COLORMAP_AUTUMN)
//             // {
//             //     double min;
//             //     double max;
//             //     cv::minMaxIdx(data, &min, &max);
//             //     cv::Mat adjMap;
//             //     // expand your range to 0..255. Similar to histEq();
//             //     data.convertTo(adjMap, CV_8UC1, 255 / (max-min), -min); 

//             //     // this is great. It converts your grayscale image into a tone-mapped one, 
//             //     // much more pleasing for the eye
//             //     // function is found in contrib module, so include contrib.hpp 
//             //     // and link accordingly
//             //     cv::Mat falseColorsMap;
//             //     applyColorMap(adjMap, falseColorsMap, colorMap);

//             //     return falseColorsMap;
//             // }

//             // cv::Mat to_8bit(int colorMap = cv::COLORMAP_AUTUMN)
//             // {
//             //     return to_8bit(img, colorMap);
//             // }

//             static void to_8bit(cv::Mat src, cv::Mat & dst, int colorMap = cv::COLORMAP_BONE)
//             {
//                 dst.release();

//                 double min, max;
//                 cv::minMaxIdx(src, &min, &max);
                
//                 cv::Mat adjMap;
//                 // expand your range to 0..255. Similar to histEq();
//                 src.convertTo(adjMap, CV_8UC1, 255 / (max-min), -min); 

//                 // this is great. It converts your grayscale image into a tone-mapped one, 
//                 // much more pleasing for the eye
//                 // function is found in contrib module, so include contrib.hpp 
//                 // and link accordingly
//                 applyColorMap(adjMap, dst, colorMap);
//             }

//             cv::Mat to_8bit(int colorMap = cv::COLORMAP_BONE)
//             {
//                 cv::Mat dst;
//                 DepthFrame::to_8bit(this->get(), dst, colorMap);
//                 return dst;
//             }

//             /*
//              * Cut out the values that fall outside [min_val, max_val] range.
//              * 
//              * @param src Source image
//              * @param max_val Maximum value for "src" elements
//              * @param min_val Minimum value for "src" elements
//              * @param new_val Value to set those elements whose value is not in the range [min_val, max_val]
//              */
//             template<typename T>
//             static void cut_at(cv::Mat src, cv::Mat & dst, float max_val, float min_val = 0, T new_val = 0)
//             {
//                 src.copyTo(dst);
//                 dst.setTo(new_val, (src < min_val) | (src > max_val));
//             }

//             template<typename T>
//             cv::Mat cut_at(float max_val, float min_val = 0, T new_val = 0)
//             {
//                 cv::Mat dst;
//                 return cut_at(this->get(), dst, max_val, min_val, new_val);
//                 return dst;
//             }
//     };

//     /*
//      * Wrapper class for color images loaded on cv::Mat structures
//      * 
//      * It provides a function for converting raw depth measurements to visually interpretable color-mapped 8-bit values.
//      */
//     class ColorFrame : public Frame
//     {
//         public:
//             ColorFrame(fs::path path, cv::Size s = cv::Size())
//               : Frame(path, s)
//             {
//                 assert(this->get_type() == CV_8UC3);
//             }

//             static void to_8bit(cv::Mat src, cv::Mat & dst)
//             {
//                 cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
//             }

//             cv::Mat to_8bit()
//             {
//                 cv::Mat dst;
//                 ColorFrame::to_8bit(this->get(), dst);
//                 return dst;
//             }
//     };

    /*
     * 
     */
    // struct Frame
    // {
    //     fs::path path;
    //     int64_t time;
    //     cv::Mat img;
    // };

    // cv::Mat read_thermal_frame(fs::path path, cv::Size s = cv::Size(640,480))
    // {
    //     /* read and preprocess frame */
    //     cv::Mat img = cv::imread(path.string(), CV_LOAD_IMAGE_UNCHANGED);
    //     img = thermal_to_8bit(img);
    //     cv::resize(img, img, s);

    //     return img;
    // }

    std::vector<int> permutation(int n)
    {
        std::vector<int> v (n);
        for (int i = 0; i < n; i++)
            v[i] = i;

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(v.begin(), v.end(), g);

        return v;
    }

    void tile(std::vector<cv::Mat> src, int tile_width, int tile_height, int grid_x, int grid_y, cv::Mat & dst) 
    {
        // patch size
        int width  =  tile_width / grid_x;
        int height = tile_height / grid_y;
        float aspect_ratio = ((float) src[0].cols) / src[0].rows;

        dst.create(tile_height, tile_width, CV_8UC3);
        dst.setTo(0);

        // iterate through grid
        int k = 0;
        for(int i = 0; i < grid_y; i++) 
        {
            for(int j = 0; j < grid_x; j++) 
            {
                cv::Mat m = src[k++];
                if (!m.empty())
                {
                    // assert(m.type() == dst.type());
                    if (m.type() != CV_8UC3)
                        cv::cvtColor(m, m, cv::COLOR_GRAY2BGR);

                    if ( (((float) m.cols) / m.rows) < aspect_ratio)
                    {
                        int new_cols = floor(m.rows * aspect_ratio);
                        int fill_size = new_cols - m.cols;
                        cv::copyMakeBorder(m, m, 0, 0, fill_size/2, fill_size/2, cv::BORDER_CONSTANT);
                    }

                    cv::resize(m, m, cv::Size(width, height));
                    m.copyTo(dst(cv::Rect(j*width, i*height, width, height))); 
                }
            }
        }
    }

    std::vector<std::string> list_images_in_directory(std::string input_dir, std::string prefix)
    {
        std::vector<std::string> files;

        fs::path input_dir_fs (input_dir);
        fs::path prefix_fs = fs::path(prefix).parent_path(); // cut out prefix of file. keep only directory prefix
        fs::directory_iterator it(input_dir_fs / prefix_fs), eod;  
        BOOST_FOREACH(const fs::path &p, std::make_pair(it, eod))   
        { 
            if(fs::is_regular_file(p) && (fs::extension(p) == ".png" || fs::extension(p) == ".jpg"))
            {
                files.push_back(p.string());
            } 
        }

        return files;
    }

    void print_mat_info(cv::Mat m)
    {
        std::cout << "Number of rows: " << m.rows << '\n';
        std::cout << "Number of columns: " << m.cols << '\n';
        std::cout << "Number of channels: " << m.channels() << '\n';
        std::cout << "Type: " << m.type() << '\n';
    }

    template <typename T>
    void mat_to_vecvec(cv::Mat m, std::vector<std::vector<T> > & vv)
    {
        vv.clear();

        for (int i = 0; i < m.rows; i++)
        {
            std::vector<T> v;
            for (int j = 0; j < m.cols; j++)
                v.push_back(m.at<T>(i,j));

            vv.push_back(v);
        }
    }

    void vec_to_mat(std::vector<cv::Point2f> points, cv::Mat & m)
    {
        m.release();

        for (cv::Point2f p : points)
        {
            m.push_back(p);
        }
    }

    cv::Mat mask_rows(cv::Mat src, cv::Mat mask)
    {
        cv::Mat dst (cv::countNonZero(mask), src.cols, src.type());

        int c = 0;
        for (int i = 0; i < src.rows; i++)
        {
            if (mask.at<unsigned char>(i,0) > 0)
            {
                src.row(i).copyTo(dst.row(c++));
            }
        }

        return dst;
    }

    bool is_bar(char c)
    {
        return c == '/' || c == '\\';
    }

    std::string type2str(int type) 
    {
        std::string r;

        uchar depth = type & CV_MAT_DEPTH_MASK;
        uchar chans = 1 + (type >> CV_CN_SHIFT);

        switch ( depth ) {
            case CV_8U:  r = "8U"; break;
            case CV_8S:  r = "8S"; break;
            case CV_16U: r = "16U"; break;
            case CV_16S: r = "16S"; break;
            case CV_32S: r = "32S"; break;
            case CV_32F: r = "32F"; break;
            case CV_64F: r = "64F"; break;
            default:     r = "User"; break;
        }

        r += "C";
        r += (chans+'0');

        return r;
    }
}

#endif /* utils_h */
