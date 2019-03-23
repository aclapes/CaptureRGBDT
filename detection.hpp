//
//  person_detection.hpp
//
//  Created by Albert Clapés on 27/11/2018.
//

#ifndef detection_h
#define detection_h

#include <opencv2/opencv.hpp>
// #include <boost/filesystem.hpp>
// #include <boost/foreach.hpp>
// #include <random>
// #include <algorithm>
// #include <iterator>
// #include <iostream>

namespace uls
{
    class MovementDetector
    {
        public:
            MovementDetector (int point_thresh, float frame_ratio) : m_point_thresh(point_thresh),
                                                                     m_frame_ratio(frame_ratio)
            {
            }

            MovementDetector (const MovementDetector & other) : m_point_thresh(other.m_point_thresh),
                                                                m_frame_ratio(other.m_frame_ratio)
            {
            }

            friend void swap (MovementDetector & first, MovementDetector & second)
            {
                using std::swap;
                swap(first.m_point_thresh, second.m_point_thresh);
                swap(first.m_frame_ratio, second.m_frame_ratio);
            }

            MovementDetector& operator=(MovementDetector other)
            {
                swap(*this, other);
                return *this;
            }

            bool find(cv::Mat src)
            {
                bool found;
                if (found = !src_prev.empty())
                {
                    assert(src.rows == src_prev.rows && src.cols == src_prev.cols && src.channels() == src_prev.channels());

                    cv::Mat src_diff;
                    cv::absdiff(src, src_prev, src_diff);
                    int nb_mv_pixels = cv::countNonZero(src_diff > m_point_thresh);
                    float r = ((float) nb_mv_pixels) / (src.rows * src.rows);
                    found = r > m_frame_ratio;
                }

                src_prev = src;
                return found;
            }

        private:
            cv::Mat src_prev;
            int m_point_thresh;
            float m_frame_ratio;

            
    };
}

#endif /* detection_h */
