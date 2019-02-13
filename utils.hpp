//
//  pt_pipeline.hpp
//  RealsenseExamplesGettingStarted
//
//  Created by Albert Clapés on 27/11/2018.
//

#ifndef utils_h
#define utils_h

#include <opencv2/opencv.hpp>

namespace utils
{
    cv::Mat thermal_to_8bit(cv::Mat data)
    {
        cv::Mat img;
        double minVal, maxVal;
        cv::Point minIdx, maxIdx;
        
        cv::normalize(data, img, 0, 65535, cv::NORM_MINMAX);
        img.convertTo(img, CV_8UC1, 1/256.);
        
        return img;
    }

    /*
     * Sam's code at https://stackoverflow.com/questions/13840013/opencv-how-to-visualize-a-depth-image
     */
    cv::Mat depth_to_8bit(cv::Mat data, int colorMap = cv::COLORMAP_AUTUMN)
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
}

#endif /* utils_h */
