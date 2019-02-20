//
//  pt_pipeline.hpp
//  RealsenseExamplesGettingStarted
//
//  Created by Albert Clapés on 27/11/2018.
//

#ifndef utils_h
#define utils_h

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

namespace fs = boost::filesystem;

namespace uls
{
    class ThermalFrame
    {
        public:
            ThermalFrame(fs::path path, cv::Size s = cv::Size(640,480))
            {
                img = cv::imread(path.string(), CV_LOAD_IMAGE_UNCHANGED);
                img = ThermalFrame::to_8bit(img);
                cv::resize(img, img, s);
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
            DepthFrame(fs::path path, cv::Size s = cv::Size(640,480))
            {
                img = cv::imread(path.string(), CV_LOAD_IMAGE_UNCHANGED);
                cv::resize(img, img, s);
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
        private:
            cv::Mat img;
    };

    class ColorFrame
    {
        public:
            ColorFrame(fs::path path) //, cv::Size s = cv::Size(640,480))
            {
                img = cv::imread(path.string(), CV_LOAD_IMAGE_UNCHANGED);
                cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
                // cv::resize(img, img, s);
            }

            cv::Mat mat() 
            {
                return img;
            }

        private:
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

    void tile(std::vector<cv::Mat> src, int tile_width, int tile_height, int grid_x, int grid_y, cv::Mat & dst) 
    {
        // patch size
        int width  =  tile_width / grid_x;
        int height = tile_height / grid_y;
        float aspect_ratio = ((float) src[0].cols) / src[0].rows;

        dst.create(tile_height, tile_width, CV_8UC3);

        // iterate through grid
        int k = 0;
        for(int i = 0; i < grid_y; i++) 
        {
            for(int j = 0; j < grid_x; j++) 
            {
                cv::Mat m = src[k++];
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

    std::vector<fs::path> list_files_in_directory(fs::path input_dir, std::string file_ext)
    {
        std::vector<fs::path> files;

        fs::directory_iterator it(input_dir), eod;  
        BOOST_FOREACH(const fs::path &p, std::make_pair(it, eod))   
        { 
            if(fs::is_regular_file(p) && fs::extension(p) == file_ext)
            {
                files.push_back(p);
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


    cv::Mat corners_2d_reference_positions(cv::Size pattern_size)
    {
        cv::Mat corners_ref (pattern_size.height*pattern_size.width, 2, CV_32FC1);

        float w_step = 1.f / (pattern_size.width-1);
        float h_step = 1.f / (pattern_size.height-1);

        for (int i = 0; i < pattern_size.height; i++)
        {
            for (int j = 0; j < pattern_size.width; j++)
            {
                corners_ref.at<float>(i*pattern_size.width+j, 0) = j*w_step;
                corners_ref.at<float>(i*pattern_size.width+j, 1) = i*h_step;
            }
        }

        return corners_ref;
    }

    bool check_corners_2d_positions(cv::Mat corners, cv::Size pattern_size, cv::Size2f eps = cv::Size2f(0.f,0.f))
    {    
        cv::Mat corners_ref = corners_2d_reference_positions(pattern_size);

        cv::Mat mask;
        cv::Mat h = cv::findHomography(corners, corners_ref, mask, CV_RANSAC);

        cv::Mat corners_transf;
        cv::perspectiveTransform(corners, corners_transf, h);

        float w_step = 1.f / (pattern_size.width-1);
        float h_step = 1.f / (pattern_size.height-1);

        if (eps.height == 0.f)
            eps.height = h_step / 4.f;
        if (eps.width == 0.f)
            eps.width = w_step / 4.f;

        for (int i = 0; i < pattern_size.height; i++)
        {
            for (int j = 0; j < pattern_size.width; j++)
            {
                float diff_x = abs( corners_transf.at<float>(i*pattern_size.width+j, 0) - j*w_step );
                float diff_y = abs( corners_transf.at<float>(i*pattern_size.width+j, 1) - i*h_step );
                if ( !(diff_x < eps.width && diff_y < eps.height) )
                    return false;
            }
        }

        return true;
    }

    /*
    * Returns true if all corners are being tracked and false otherwise (some of them are lost)
    */
    bool check_corners_integrity(cv::Mat corners_status, cv::Size pattern_size)
    {
        return pattern_size.width * pattern_size.height == cv::sum(corners_status)[0];
    }

    static void calcBoardCornerPositions(cv::Size pattern_size, float square_width, float square_height, std::vector<cv::Point3f>& corners)
    {
        corners.clear();
        
        for( int i = 0; i < pattern_size.height; ++i )
            for( int j = 0; j < pattern_size.width; ++j )
                corners.push_back(cv::Point3f(float( j*square_width ), float( i*square_height ), 0));
    }

    template<typename T>
    void find_chessboard_corners(std::vector<fs::path> frames, 
                                 cv::Size pattern_size, 
                                 std::vector<cv::Mat> & frames_corners,
                                 std::vector<int> & frames_inds,
                                 bool verbose = true) 
    {
        frames_corners.clear();
        frames_inds.clear(); 

        cv::Mat img, img_prev;
        cv::Mat corners, corners_prev;
        float tracking_enabled = false;       

        for (int i = 0; i < frames.size(); i++) 
        {
            /* read and preprocess frame */
            cv::Mat img = T(frames[i]).mat();

            corners.release();
            bool chessboard_found = findChessboardCorners(img, pattern_size, corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
            
            if (chessboard_found) 
            {
                cornerSubPix(img, corners, cv::Size(15, 15), cv::Size(5, 5), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
                tracking_enabled = true;
            }
            else if (tracking_enabled)
            {
                cv::Mat status, err;
                cv::calcOpticalFlowPyrLK(img_prev, img, corners_prev, corners, status, err, cv::Size(7,7));
                cornerSubPix(img, corners, cv::Size(15, 15), cv::Size(5, 5), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
                // error checking
                if ( !check_corners_integrity(status, pattern_size) || !check_corners_2d_positions(corners, pattern_size) )
                {
                    tracking_enabled = false;
                    corners.release();
                }
            }

            if (!corners.empty()) 
            {
                frames_corners.push_back(corners);
                frames_inds.push_back(i);
            }

            if (verbose)
            {
                cv::Mat cimg;
                cv::cvtColor(img, cimg, cv::COLOR_GRAY2BGR);
                if (!corners.empty()) cv::drawChessboardCorners(cimg, pattern_size, corners, chessboard_found);
                cv::imshow("Viewer", cimg);
                cv::waitKey(1);
            }

            img_prev = img;
            corners_prev = corners;
        }
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
