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

    struct Timestamp
    {
        std::string id;
        int64_t time;
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

    cv::Mat orient_corners(cv::Mat src)
    {
        cv::Mat dst;
        cv::Point2f pi, pf;

        pi = src.at<cv::Point2f>(0,0);
        pf = src.at<cv::Point2f>(0,src.cols-1);
        if (pi.x < pf.x && pi.y < pf.y)
            return src;
        else
        {
            cv::Mat oriented;
            cv::flip(src, oriented, 1);
            return oriented;
        }             
    }

    std::vector<std::string> list_files_in_directory(std::string input_dir, std::string prefix, std::string file_ext)
    {
        std::vector<std::string> files;

        fs::path input_dir_fs (input_dir);
        fs::directory_iterator it(input_dir_fs), eod;  
        BOOST_FOREACH(const fs::path &p, std::make_pair(it, eod))   
        { 
            if(fs::is_regular_file(p) && fs::extension(p) == file_ext)
            {
                files.push_back(p.string());
            } 
        }

        return files;
    }

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
                try {
                    uls::Timestamp ts = process_log_line(line);
                    tokenized_lines.push_back(ts);
                }
                catch (std::exception & e)
                {
                    break;
                }
            }
            log.close();
        }

        return tokenized_lines;
    }

    void print_mat_info(cv::Mat m)
    {
        std::cout << "Number of rows: " << m.rows << '\n';
        std::cout << "Number of columns: " << m.cols << '\n';
        std::cout << "Number of channels: " << m.channels() << '\n';
        std::cout << "Type: " << m.type() << '\n';
    }

    void align_pattern_corners(cv::Mat a, cv::Mat b, cv::Size pattern_a, cv::Size pattern_b, cv::Mat & aa, cv::Mat & bb, cv::Size & pattern)
    {
        cv::Mat a_ = a.reshape(a.channels(), pattern_a.height);
        cv::Mat b_ = b.reshape(b.channels(), pattern_b.height);

        int min_rows = std::min(a_.rows, b_.rows);
        int min_cols = std::min(a_.cols, b_.cols);

        aa = a_(cv::Rect((a_.cols-min_cols)/2, (a_.rows-min_rows)/2, min_cols, min_rows));
        bb = b_(cv::Rect((b_.cols-min_cols)/2, (b_.rows-min_rows)/2, min_cols, min_rows));

        aa = aa.clone().reshape(a_.channels(), 1);
        bb = bb.clone().reshape(b_.channels(), 1);

        pattern.height = min_rows;
        pattern.width = min_cols;
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
        cv::Mat h = cv::findHomography(corners, corners_ref, mask, cv::RANSAC);

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
    void find_chessboard_corners(std::vector<std::string> frames, 
                                 cv::Size pattern_size, 
                                 std::vector<cv::Mat> & frames_corners,
                                 std::vector<int> & frames_inds,
                                 cv::Size resize_dims = cv::Size(),
                                 std::string prefix = "",
                                 int y_shift = 0,
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
            cv::Mat img = T(fs::path(prefix) / fs::path(frames[i]), resize_dims, y_shift).mat();

            corners.release();
            // cv::GaussianBlur(fra, img, cv::Size(0, 0), 3);
            // cv::addWeighted(fra, 1.5, img, -0.5, 0, img);
            bool chessboard_found = findChessboardCorners(img, pattern_size, corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
            
            if (chessboard_found) 
            {
                cornerSubPix(img, corners, cv::Size(21, 21), cv::Size(7, 7), cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 1e-1));
                tracking_enabled = true;
            }
            else if (tracking_enabled)
            {
                cv::Mat status, err;
                cv::calcOpticalFlowPyrLK(img_prev, img, corners_prev, corners, status, err, cv::Size(7,7));
                cornerSubPix(img, corners, cv::Size(21, 21), cv::Size(7, 7), cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 1e-1));
                // error checking
                if ( ! check_corners_integrity(status, pattern_size) )
                {
                    tracking_enabled = false;
                    corners.release();
                }
            }

            if (tracking_enabled)
            {
                if ((corners.rows == pattern_size.width * pattern_size.height) && check_corners_2d_positions(corners, pattern_size))
                {
                    frames_corners.push_back(corners);
                    frames_inds.push_back(i);
                }
                else
                {
                    corners.release();
                    tracking_enabled = false;
                }
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

    void _transform_point_domain(cv::Mat src, cv::Size dom_src, cv::Size dom_dst, cv::Mat & dst)
    {
        dst.release();
        dst.create(src.rows, src.cols, src.type());

        float ratio_1 = ((float) dom_dst.width) / dom_dst.height;
        float ratio_2 = ((float) dom_src.width) / dom_src.height;

        cv::Size new_domain;
        cv::Size offset;

        if (ratio_2 < ratio_1)
        {
            new_domain = cv::Size(dom_dst.height * ratio_2, dom_dst.height);
            offset = cv::Size((dom_dst.width - dom_dst.height * ratio_2)/2., 0);
        }
        else
        {
            new_domain = cv::Size(dom_dst.width, dom_dst.width / ratio_2);
            offset = cv::Size(0, (dom_dst.height - dom_dst.width / ratio_2)/2.);
        }

        for (int i = 0; i < src.rows; i++) 
        {
            for (int j = 0; j < src.cols; j++)
            {
                cv::Point2f p = src.at<cv::Point2f>(i,j);
                float p_x = (p.x / dom_src.width) * new_domain.width;
                float p_y = (p.y / dom_src.height) * new_domain.height;
                dst.at<cv::Point2f>(i,j) = cv::Point2f(p_x + offset.width, p_y + offset.height);
            }
        }
    }

    void transform_point_domains(cv::Mat points_1, cv::Mat points_2, cv::Size dom_1, cv::Size dom_2, cv::Mat & points_1_transf, cv::Mat & points_2_transf, cv::Size & dom_transf)
    {
        points_1_transf.release();
        points_2_transf.release();

        dom_transf = cv::Size (std::max(dom_1.width, dom_2.width), std::max(dom_1.height, dom_2.height));
        _transform_point_domain(points_1, dom_1, dom_transf, points_1_transf);
        _transform_point_domain(points_2, dom_2, dom_transf, points_2_transf);
    }

    void time_sync(std::vector<Timestamp> log_a, std::vector<Timestamp> log_b, std::vector<std::pair<Timestamp,Timestamp> > & log_synced, int64_t eps = 50, bool verbose = true)
    {
        std::vector<Timestamp> master;
        std::vector<Timestamp> slave;
        if (log_a.size() > log_b.size())
        {
            master = log_a;
            slave  = log_b;
            if (verbose) std::cout << "a is master, b is slave\n";
        } else {
            master = log_b;
            slave  = log_a;
            if (verbose) std::cout << "b is master, a is slave\n";
        }

        int j = 0;

        for (int i = 0; i < master.size(); i++) 
        {
            Timestamp ts_m = master[i];
            std::vector<std::pair<Timestamp, int64_t> > matches;
            while (j < slave.size() && slave[j].time < ts_m.time + eps)
            {
                int64_t dist = abs(ts_m.time - slave[j].time);
                if (dist < eps)
                {
                    matches.push_back( std::pair<Timestamp,int64_t>(slave[j], dist) );
                }
                j++;
            }

            if (!matches.empty())
            {
                std::pair<Timestamp, int64_t> m_best = matches[0];
                for (int k = 1; k < matches.size(); k++)
                {
                    if (matches[k].second < m_best.second)
                        m_best = matches[k];
                }

                std::pair<Timestamp,Timestamp> synced_pair;
                synced_pair.first  = log_a.size() > log_b.size() ? master[i] : m_best.first;
                synced_pair.second = log_a.size() > log_b.size() ? m_best.first : master[i];
                log_synced.push_back(synced_pair);
            }
            // elif fill_with_previous:
            //     all_matches.append( ((i, ts_m), all_matches[-1][1]) ), ts_m in enumerate(master)
        }

        // for (auto log : log_synced)
        // {
        //     std::cout << log.first.time << "," << log.second.time << '\n';
        // }

        if (verbose)
        {
            for (int i = 0; i < log_synced.size(); i++)
            {
                std::cout << log_synced[i].first.time << "," << log_synced[i].second.time << '\n';
            }
        }
    }

    void time_sync(std::vector<Timestamp> log_a, std::vector<Timestamp> log_b, std::vector<std::pair<int,int> > & log_pairs, int64_t eps = 50, bool verbose = true)
    {
        log_pairs.clear();

        // std::vector<std::pair<Timestamp,Timestamp> > log_synced;

        std::vector<Timestamp> master;
        std::vector<Timestamp> slave;
        if (log_a.size() > log_b.size())
        {
            master = log_a;
            slave  = log_b;
            if (verbose) std::cout << "a is master, b is slave\n";
        } else {
            master = log_b;
            slave  = log_a;
            if (verbose) std::cout << "b is master, a is slave\n";
        }

        int j = 0;

        for (int i = 0; i < master.size(); i++) 
        {
            Timestamp ts_m = master[i];
            std::vector<std::pair<int, int64_t> > matches;
            while (j < slave.size() && slave[j].time < ts_m.time + eps)
            {
                int64_t dist = abs(ts_m.time - slave[j].time);
                if (dist < eps)
                {
                    matches.push_back( std::pair<int, int64_t>(j, dist) );
                }
                j++;
            }

            if (!matches.empty())
            {
                std::pair<int,int64_t> m_best = matches[0];
                for (int k = 1; k < matches.size(); k++)
                {
                    if (matches[k].second < m_best.second)
                        m_best = matches[k];
                }
                
                std::pair<int,int> synced_pair;
                synced_pair.first  = log_a.size() > log_b.size() ? i : m_best.first;
                synced_pair.second = log_a.size() > log_b.size() ? m_best.first : i;
                log_pairs.push_back(synced_pair);
            }
        }

        // if (verbose)
        // {
        //     for (int i = 0; i < log_synced.size(); i++)
        //     {
        //         std::cout << log_synced[i].first.time << "," << log_synced[i].second.time << '\n';
        //     }
        // }
    }
}

#endif /* utils_h */
