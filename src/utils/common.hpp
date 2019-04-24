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
     * Wrapper to cv::resize with automatic padding to keep aspect ratio
     */
    void resize(cv::Mat src, cv::Mat & dst, cv::Size s)
    {
        if (s.empty() || (src.rows == s.height && src.cols == s.width))
            src.copyTo(dst);
        else
        {
            float ratio_1 = ((float) s.width) / s.height;
            float ratio_2 = ((float) src.cols) / src.rows;

            cv::Size new_domain, offset;

            if (ratio_2 < ratio_1)
            {
                new_domain = cv::Size(s.height * ratio_2, s.height);
                offset = cv::Size((s.width - new_domain.height * ratio_2)/2., 0);
            }
            else
            {
                new_domain = cv::Size(new_domain.width, new_domain.width / ratio_2);
                offset = cv::Size(0, (s.height - new_domain.width / ratio_2)/2.);
            }

            cv::resize(src, dst, new_domain);
            cv::copyMakeBorder(dst, dst, 
                                offset.height, offset.height,
                                offset.width,  offset.width, 
                                cv::BORDER_CONSTANT);
        }
    }

    class ThermalFrame
    {
        public:
            ThermalFrame(fs::path path, cv::Size s = cv::Size(), int y_shift = 0)
            {
                img = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
                img = ThermalFrame::to_8bit(img);
                resize(img, img, s);
                if (y_shift > 0)
                    cv::copyMakeBorder(img, img, 0, y_shift, 0, 0, cv::BORDER_CONSTANT);
                else if (y_shift < 0)
                    cv::copyMakeBorder(img, img, abs(y_shift), 0, 0, 0, cv::BORDER_CONSTANT);
            }

            static cv::Mat to_8bit(cv::Mat data)
            {
                cv::Mat img;
                double minVal, maxVal;
                cv::Point minIdx, maxIdx;
                
                cv::normalize(data, img, 0, 65535, cv::NORM_MINMAX);
                img.convertTo(img, CV_8UC1, 1/256.);
                
                return img;
            }

            cv::Mat mat() 
            {
                return img;
            }

        private:
            cv::Mat img;
    };

    class DepthFrame
    {
        public:
            DepthFrame(fs::path path, cv::Size s = cv::Size(), int y_shift = 0)
            {
                img = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
                resize(img, img, s);
                if (y_shift > 0)
                    cv::copyMakeBorder(img, img, 0, y_shift, 0, 0, cv::BORDER_CONSTANT);
                else if (y_shift < 0)
                    cv::copyMakeBorder(img, img, abs(y_shift), 0, 0, 0, cv::BORDER_CONSTANT);
            }

            cv::Mat mat() 
            {
                return img;
            }
            /*
            * Sam's code at https://stackoverflow.com/questions/13840013/opencv-how-to-visualize-a-depth-image
            */
            static cv::Mat to_8bit(cv::Mat data, int colorMap = cv::COLORMAP_AUTUMN)
            {
                double min;
                double max;
                cv::minMaxIdx(data, &min, &max);
                cv::Mat adjMap;
                // expand your range to 0..255. Similar to histEq();
                data.convertTo(adjMap, CV_8UC1, 255 / (max-min), -min); 

                // this is great. It converts your grayscale image into a tone-mapped one, 
                // much more pleasing for the eye
                // function is found in contrib module, so include contrib.hpp 
                // and link accordingly
                cv::Mat falseColorsMap;
                applyColorMap(adjMap, falseColorsMap, colorMap);

                return falseColorsMap;
            }

            cv::Mat to_8bit(int colorMap = cv::COLORMAP_AUTUMN)
            {
                return to_8bit(img, colorMap);
            }

            template<typename T>
            static cv::Mat cut_at(cv::Mat src, float max_z, float min_z = 0, T val = 0)
            {
                cv::Mat dst = src.clone();
                dst.setTo(val, (src > max_z) | (src < min_z));


                return dst;
            }

        private:
            cv::Mat img;
    };

    class ColorFrame
    {
        public:
            ColorFrame(fs::path path, cv::Size s = cv::Size(), int y_shift = 0)
            {
                img = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
                cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
                resize(img, img, s);
                if (y_shift > 0)
                    cv::copyMakeBorder(img, img, 0, y_shift, 0, 0, cv::BORDER_CONSTANT);
                else if (y_shift < 0)
                    cv::copyMakeBorder(img, img, abs(y_shift), 0, 0, 0, cv::BORDER_CONSTANT);
            }

            cv::Mat mat() 
            {
                return img;
            }

        private:
            cv::Mat img;
    };

    struct Frame
    {
        fs::path path;
        int64_t time;
        cv::Mat img;
    };

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

    std::vector<std::string> list_files_in_directory(std::string input_dir, std::string prefix, std::string file_ext)
    {
        std::vector<std::string> files;

        fs::path input_dir_fs (input_dir);
        fs::path prefix_fs = fs::path(prefix).parent_path(); // cut out prefix of file. keep only directory prefix
        fs::directory_iterator it(input_dir_fs / prefix_fs), eod;  
        BOOST_FOREACH(const fs::path &p, std::make_pair(it, eod))   
        { 
            if(fs::is_regular_file(p) && fs::extension(p) == file_ext)
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
}

#endif /* utils_h */
