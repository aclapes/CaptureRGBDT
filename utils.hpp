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
}

#endif /* utils_h */
