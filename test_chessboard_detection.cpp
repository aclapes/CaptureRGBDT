#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/utils/logger.defines.hpp>
#include <opencv2/core/utils/logger.hpp>

void align_pattern_corners(cv::Mat a, cv::Mat b, cv::Size pattern_a, cv::Size pattern_b, cv::Mat & aa, cv::Mat & bb)
{
    cv::Mat a_ = a.reshape(a.channels(), pattern_a.height);
    cv::Mat b_ = b.reshape(b.channels(), pattern_b.height);

    int min_rows = std::min(a_.rows, b_.rows);
    int min_cols = std::min(a_.cols, b_.cols);

    aa = a_(cv::Rect((a_.cols-min_cols)/2, (a_.rows-min_rows)/2, min_cols, min_rows));
    bb = b_(cv::Rect((b_.cols-min_cols)/2, (b_.rows-min_rows)/2, min_cols, min_rows));

    aa = aa.clone().reshape(a_.channels(), 1);
    bb = bb.clone().reshape(b_.channels(), 1);
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

void transform_point_domains(cv::Mat points_1, cv::Mat points_2, cv::Size dom_1, cv::Size dom_2, cv::Mat & points_1_transf, cv::Mat & points_2_transf)
{
    points_1_transf.release();
    points_2_transf.release();

    cv::Size dom_merged (std::max(dom_1.width, dom_2.width), std::max(dom_1.height, dom_2.height));
    _transform_point_domain(points_1, dom_1, dom_merged, points_1_transf);
    _transform_point_domain(points_2, dom_2, dom_merged, points_2_transf);
}



// void foo(cv::Mat points_1, cv::Mat points_2, cv::Size domain_1, cv::Size domain_2, cv::Mat & points_transformed)
// {
//     points_transformed.release();

//     if (domain_1.width * domain_1.height > domain_2.width * domain_2.height)
//         _foo(points_1, points_2, domain_1, domain_2, points_transformed);
//     else
//         _foo(points_2, points_1, domain_2, domain_1, points_transformed);
// }

int main(int argc, char * argv[])
{    
    // int max_cols = 1280;
    // int max_rows = 720;
    // cv::Size frame_size (640, 480);
    // cv::Mat a(1,1,CV_32FC2,cv::Scalar(640,480));
    // std::cout << a << std::endl;
    // cv::multiply(a, cv::Scalar(((float) max_cols)/frame_size.width, ((float) max_rows)/frame_size.height), a);
    // std::cout << a << std::endl;
    // cv::Mat a, b;
    // a = (cv::Mat_<int>(5,5) <<  8, 1, 2, 5, 2,
    //                             9, 0, 1, 0, 0,
    //                             9, 1, 1, 3, 0,
    //                             7, 0, 0, 1, 4,
    //                             1, 4, 1, 4, 4);

    cv::Mat A = (cv::Mat_<float>(3,3) <<  8, 8, 8,
                                1, 0, 2,
                                8, 1, 2);
    cv::Mat B = (cv::Mat_<float>(3,3) <<  8, 8, 8,
                                1, 0, 2,
                                8, 1, 2);
    
    cv::Mat C;
    C = A * B;
    
    cv::Mat a (1, 2, CV_32FC2);
    a.at<cv::Point2f>(0,0) = cv::Point2f(0,0);
    a.at<cv::Point2f>(0,1) = cv::Point2f(1280,720);
    std::cout << a <<std::endl;

    cv::Mat b (1, 2, CV_32FC2);
    b.at<cv::Point2f>(0,0) = cv::Point2f(0,0);
    b.at<cv::Point2f>(0,1) = cv::Point2f(1440,480);
    std::cout << b <<std::endl;

    cv::Mat a_transf, b_transf;
    transform_point_domains(a, b, cv::Size(1280,720), cv::Size(1440,480), a_transf, b_transf);
    std::cout << a_transf << std::endl;
    std::cout << b_transf << std::endl;
    // a = a.reshape(1, 1);
    // b = b.reshape(1, 1);
    // align_pattern_corners(b,a, cv::Size(3,3), cv::Size(5,5), bb, aa);

    // std::cout << aa << std::endl;
    // std::cout << bb << std::endl;

    // cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_DEBUG);
    // cv::Mat img = cv::imread("./chessboard/t_00000146.png", CV_LOAD_IMAGE_UNCHANGED);

    // // img.setTo(10000, img < 10000);
    // // img.setTo(50000, img > 50000);

    // cv::resize(img, img, cv::Size(1280,960));
    // // cv::Mat blur;
    // // cv::GaussianBlur(img, blur, cv::Size(0, 0), 3);
    // // cv::addWeighted(img, 1.5, blur, -0.5, 0, img);

    // double minVal, maxVal;
    // cv::Point minIdx, maxIdx;

    // cv::normalize(img, img, 0, 65535, cv::NORM_MINMAX);
    // img.convertTo(img, CV_8UC1, 1/256.);
    // // img = 255 - img;
    // // cv::equalizeHist(img, img);

    // cv::Mat corners;
    // bool found = cv::findChessboardCorners(img, cv::Size(9,6), corners, cv::CALIB_CB_ADAPTIVE_THRESH);
    // cv::drawChessboardCorners(img, cv::Size(9,6), corners, found);
    // std::cout << found << std::endl;

    // while (true)
    // {
    //     cv::imshow("Viewer", img);
    //     char key = cv::waitKey();
    //     if (key == 27)
    //         break;
    // }

    return 0;
}