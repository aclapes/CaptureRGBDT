#include <opencv2/opencv.hpp>
#include <iostream>
#include "../utils/common.hpp"
#include <boost/filesystem.hpp>

// cv::Rect resize(cv::Mat src, cv::Mat & dst, cv::Size s)
// {
//     cv::Rect roi;

//     if (s.empty() || (src.rows == s.height && src.cols == s.width))
//     {
//         src.copyTo(dst);
//         roi = cv::Rect(0,0,dst.cols,dst.cols);
//     }
//     else
//     {
//         float ratio_s = ((float) s.width) / s.height;
//         float ratio_src = ((float) src.cols) / src.rows;

//         cv::Size new_domain, offset;

//         if (ratio_src == ratio_s)
//         {
//             cv::resize(src, dst, s);
//         }
//         else if (ratio_src < ratio_s)
//         {
//             new_domain = cv::Size(s.height * ratio_src, s.height);
//             offset = cv::Size((s.width - new_domain.height * ratio_src)/2., 0);
//         }
//         else
//         {
//             new_domain = cv::Size(s.width, s.width / ratio_src);
//             offset = cv::Size(0, (s.height - new_domain.width / ratio_src)/2.);
//         }

//         cv::Mat tmp;
//         cv::resize(src, tmp, new_domain);
//         cv::copyMakeBorder(tmp, dst, 
//                             offset.height, offset.height,
//                             offset.width,  offset.width, 
//                             cv::BORDER_CONSTANT);

//         roi = cv::Rect(offset.width, offset.height, new_domain.width, new_domain.height);
//     }

//     return roi;
// }
//
// void to_8bit(cv::Mat src, cv::Mat & dst)
// {
//     cv::Mat tmp;
//     cv::normalize(src, tmp, 0, 65535, cv::NORM_MINMAX, CV_16UC1);
//     tmp.convertTo(dst, CV_8UC1, 1/256.);
// }

int main(int argc, char * argv[])
{   
    std::cout << "hello resize" << std::endl;
    // uls::ThermalFrame f (boost::filesystem::path("/home/aclapes/Code/c++/getting-started/data/people-green/2019-04-20_15.36.42/pt/thermal/t_00000036.png"));
    // f.resize(cv::Size(400,240));
    // cv::imshow("foo", f.to_8bit());
    // cv::waitKey();

    // cv::Mat img = cv::imread("/home/aclapes/Code/c++/getting-started/data/people-green/2019-04-20_15.36.42/pt/thermal/t_00000036.png", cv::IMREAD_UNCHANGED);
    // cv::Mat dst;
    // cv::Rect roi;
    // cv::Mat tmp;

    // roi = resize(img, dst, cv::Size(1280,720)); // OK
    // // cv::rectangle(dst, roi, cv::Scalar(0,0,255), 1, 8);
    // cv::normalize(dst(roi), dst(roi), 0, 65535, cv::NORM_MINMAX, CV_16UC1);
    // // tmp.copyTo(dst(roi));
    // dst.convertTo(dst, CV_8UC1, 1/256.);
    // cv::imshow("img", dst);
    // cv::waitKey();

    // roi = resize(img, dst, cv::Size(1280,480)); // OK
    // cv::rectangle(dst, roi, cv::Scalar(0,0,255), 1, 8);
    // cv::imshow("img", dst);
    // cv::waitKey();

    // roi = resize(img, dst, cv::Size(1920,720)); // OK
    // cv::rectangle(dst, roi, cv::Scalar(0,0,255), 1, 8);
    // cv::imshow("img", dst);
    // cv::waitKey();

    // roi = resize(img, dst, cv::Size(480,900)); // OK
    // cv::rectangle(dst, roi, cv::Scalar(0,0,255), 1, 8);
    // cv::imshow("img", dst);
    // cv::waitKey();
    return 0;
}