//
// calibration.hpp
// 
//  Created by Albert Clapés on 16/04/2018.
//

#ifndef calibration_hpp
#define calibration_hpp

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>

#include "common.hpp"

namespace fs = boost::filesystem;

namespace uls
{
    /* Struct definitions */
    typedef struct
    {
        cv::Mat camera_matrix;
        cv::Mat dist_coeffs;
    } intrinsics_t;

    typedef struct
    {
        cv::Mat R;
        cv::Mat T;
    } extrinsics_t;

    /*
     * It transforms the domain of src 2D points to a newer domain.
     * 
     * Auxiliary function of homogeneize_2d_domains.
     * 
     * @src Set of 2D points
     * @dom_src Original domain of the 2D points
     * @dom_dst Newer domain size
     * @dst Output set of transformed 2D points
     */
    void _homogeneize_2d_domain(cv::Mat src, cv::Size dom_src, cv::Size dom_dst, cv::Mat & dst)
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

    /*
     * Given two sets of points each in a different domain, it homogeneizes their domains given a criterion either "max" (default) or "min".
     * 
     * The 2D domains of the two sets of points_1 and points_2 are homogeneized to the same domain, either dom_1 or dom_2. This is done
     * by considering each domain's dimension size and transforming the dimension of the correspoing set of points to the domains' largest dimension
     * when criterion = "max" or to the smallest dimension when criterion = "min".
     * Check _homogeneize_2d_domain helper function for more details.
     * 
     * @points_1 First set of points
     * @points_2 Second set of points
     * @dom_1 2D domain of points_1
     * @dom_2 2D domain of points_2
     * @points_1_transf Output transformation of points_1 transformed to new domain given the Homogeneization criterion
     * @points_2_transf Output transformation of points_2 transformed to new domain given the Homogeneization criterion
     * @criterion Homogeneization criterion, either "max" (default) or "min"
     */
    void homogeneize_2d_domains(cv::Mat points_1, 
                                 cv::Mat points_2, 
                                 cv::Size dom_1,
                                 cv::Size dom_2, 
                                 cv::Mat & points_1_transf, 
                                 cv::Mat & points_2_transf, 
                                 cv::Size & dom_transf,
                                 std::string criterion = "max")
    {
        points_1_transf.release();
        points_2_transf.release();

        if (criterion == "max")
            dom_transf = cv::Size (std::max(dom_1.width, dom_2.width), std::max(dom_1.height, dom_2.height));
        else if (criterion == "min")
            dom_transf = cv::Size (std::min(dom_1.width, dom_2.width), std::min(dom_1.height, dom_2.height));
        else
            throw std::runtime_error("transform_point_domain(...): not a valid criterion");

        _homogeneize_2d_domain(points_1, dom_1, dom_transf, points_1_transf);
        _homogeneize_2d_domain(points_2, dom_2, dom_transf, points_2_transf);
    }

    /*
     * Deconstructs a camera matrix into the set of 4 intrinsic parameters.
     * @param K The camera matrix
     * @param fx Output focal distance along x axis
     * @param fy Output focal distance along y axis
     * @param cx Output principal image point x coordinate
     * @param cy Output principal image point y coordinate
     */
    void camera_matrix_to_intrinsics(cv::Mat K, double & fx, double & fy, double & cx, double & cy)
    {
        fx = K.at<double>(0,0);
        fy = K.at<double>(1,1);
        cx = K.at<double>(0,2);
        cy = K.at<double>(1,2);
    }

    /*
     * Constructs a camera matrix from the set of 4 intrinsic parameters.
     * @param fx Focal distance along x axis
     * @param fy Focal distance along y axis
     * @param cx Principal image point x coordinate
     * @param cy Principal image point y coordinate
     * @param K Output camera matrix
     */
    void intrinsics_to_camera_matrix(double fx, double fy, double cx, double cy, cv::Mat & K)
    {
        K = (cv::Mat_<double>(3, 3, CV_64F) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        // K.release();
        // K.create(3, 3, CV_64FC1);
        // K.setTo(0);

        // K.at<double>(0,0) = fx;
        // K.at<double>(1,1) = fy;
        // K.at<double>(0,2) = cx;
        // K.at<double>(1,2) = cy; 
    }

    /*
     * Computes the inverse mapping transformation in order to align some "other" modality to the depth modality given a set of input parameters.
     *
     * @param depth The depth image
     * @param K_depth The depth camera matrix
     * @param K_other The other mdality camera matrix
     * @param depth_scale The scaling factor to transform the depth image pixel values into meters (not sure if this is needed)
     * @param extrinsics The rotation and translation of the other camera with respect to the depth one
     * @param Output x coordinates' inverse mapping matrix
     * @param Output y coordinates' inverse mapping matrix
     */
    void align_to_depth(cv::Mat depth, 
                        cv::Mat K_depth,
                        cv::Mat K_other,
                        float depth_scale,
                        std::shared_ptr<extrinsics_t> extrinsics,
                        cv::Mat & map_x, 
                        cv::Mat & map_y)
    {
        double fx_d, fy_d, cx_d, cy_d, fx_o, fy_o, cx_o, cy_o;
        uls::camera_matrix_to_intrinsics(K_depth, fx_d, fy_d, cx_d, cy_d);
        uls::camera_matrix_to_intrinsics(K_other, fx_o, fy_o, cx_o, cy_o);

        cv::Mat R = extrinsics->R;
        cv::Mat T = extrinsics->T;

        double x, y, z;
        double p_x, p_y, p_z;

        map_x.release();
        map_y.release();

        map_x.create(depth.size(), CV_32FC1);
        map_y.create(depth.size(), CV_32FC1);

        map_x.setTo(-1);
        map_y.setTo(-1);

        for (int i = 0; i < depth.rows; i++)
        {
            for (int j = 0; j < depth.cols; j++)
            {
                z = depth_scale * depth.at<unsigned short>(i,j);
                if (z > 0)
                {
                    x = (j - cx_d) * z / fx_d;
                    y = (i - cy_d) * z / fy_d;

                    p_x = (R.at<double>(0,0) * x + R.at<double>(0,1) * y + R.at<double>(0,2) * z) + T.at<double>(0,0);
                    p_y = (R.at<double>(1,0) * x + R.at<double>(1,1) * y + R.at<double>(1,2) * z) + T.at<double>(1,0);
                    p_z = (R.at<double>(2,0) * x + R.at<double>(2,1) * y + R.at<double>(2,2) * z) + T.at<double>(2,0);

                    map_x.at<float>(i,j) = (p_x * fx_o / p_z) + cx_o;
                    map_y.at<float>(i,j) = (p_y * fy_o / p_z) + cy_o;
                }
            }
        }
    }

    /*
    * Performs chessboard detection on an image given a pattern_size and draws it on top of the image.
    * 
    * @param img Image where to find the chessboard corners
    * @param pattern_size Size of the chessboard pattern
    * @flags Pattern detection flags (see cv::findChessboardCorners's "flags" parameter values)
    * @return
    */
    void find_and_draw_chessboard(cv::Mat & img, cv::Size pattern_size, int flags = 0)
    {
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        cv::Mat corners;
        bool found = cv::findChessboardCorners(gray, pattern_size, corners, flags);
        
        if (img.type() != CV_8UC3)
            cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

        cv::drawChessboardCorners(img, pattern_size, corners, found);
    }

    /*
     * Orients corners of a detected chessboard pattern
     * 
     * It might happen the corners are detected starting from the right-bottom corner to the top-left instead of top-left to right-bottom.
     * If so, change the order of the list of detected corners to the second orientation, from top-left -> right-bottom.
     * 
     * @param src List of 2D corner points
     */
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

    /*
     * Computes the intersection of different-sized patterns.
     * 
     * @param a Corners of the first pattern
     * @param b Corners of the second pattern
     * @param pattern_a Size of pattern a
     * @param pattern_b size of pattern b
     * @param aa Output corners of a that fall in the pattern intersection
     * @param bb Output corners of b that fall in the pattern intersection
     */
    void intersect_patterns(cv::Mat a, cv::Mat b, cv::Size pattern_a, cv::Size pattern_b, cv::Mat & aa, cv::Mat & bb, cv::Size & pattern)
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

    // THIS IS NOT AND INDEPENDENT FUNCTION ANYMORE ( INCLUDED IN check_corners_2d_positions(...) )
    // cv::Mat corners_2d_reference_positions(cv::Size pattern_size)
    // {
    //     cv::Mat corners_ref (pattern_size.height*pattern_size.width, 2, CV_32FC1);

    //     float w_step = 1.f / (pattern_size.width-1);
    //     float h_step = 1.f / (pattern_size.height-1);

    //     for (int i = 0; i < pattern_size.height; i++)
    //     {
    //         for (int j = 0; j < pattern_size.width; j++)
    //         {
    //             corners_ref.at<float>(i*pattern_size.width+j, 0) = j*w_step;
    //             corners_ref.at<float>(i*pattern_size.width+j, 1) = i*h_step;
    //         }
    //     }

    //     return corners_ref;
    // }

    /*
     * Generates a list of corners 2d positions
     * 
     * @param Size of the pattern
     * @param square_width Width of the pattern squares
     * @param square_height Height of the pattern squares
     * @parma Output of corner positions
     */
    static void calcBoardCornerPositions(cv::Size pattern_size, float square_width, float square_height, std::vector<cv::Point2f>& corners)
    {
        corners.clear();
        
        for( int i = 0; i < pattern_size.height; ++i )
            for( int j = 0; j < pattern_size.width; ++j )
                corners.push_back(cv::Point2f(float( j*square_width ), float( i*square_height )));
    }

    /*
     * Generates a list of corners 3d positions
     * 
     * The listed corners have z = 0. This function is used in quite a few OpenCV calibration functions.
     * 
     * @param Size of the pattern
     * @param square_width Width of the pattern squares
     * @param square_height Height of the pattern squares
     * @parma Output of corner positions
     */
    static void calcBoardCornerPositions(cv::Size pattern_size, float square_width, float square_height, std::vector<cv::Point3f>& corners)
    {
        corners.clear();
        
        for( int i = 0; i < pattern_size.height; ++i )
            for( int j = 0; j < pattern_size.width; ++j )
                corners.push_back(cv::Point3f(float( j*square_width ), float( i*square_height ), 0));
    }

    /*
     * Checks the grid-like structure supposedly formed by a set of detected corners.
     * 
     * It transforms "corners" to the ideal corner diposition according to the pattern shape and checks the grid-like
     * structure of "corners" is mantained.
     * 
     * @param corners Detected set of corners
     * @param pattern_size Size of the corners's pattern
     * @param eps A threshold to check the deviation of the detected corners w.r.t to the ideal grid-like structure of the pattern
     * @returns 
     */
    bool check_corners_2d_positions(cv::Mat corners, cv::Size pattern_size, cv::Size2f eps = cv::Size2f(0.f,0.f))
    {    
        // cv::Mat corners_ref (pattern_size.height*pattern_size.width, 2, CV_32FC1);

        // float w_step = 1.f / (pattern_size.width-1);
        // float h_step = 1.f / (pattern_size.height-1);

        // for (int i = 0; i < pattern_size.height; i++)
        // {
        //     for (int j = 0; j < pattern_size.width; j++)
        //     {
        //         corners_ref.at<float>(i*pattern_size.width+j, 0) = j*w_step;
        //         corners_ref.at<float>(i*pattern_size.width+j, 1) = i*h_step;
        //     }
        // }

        std::vector<cv::Point2f> corners_tmp;
        calcBoardCornerPositions(pattern_size, 1.f/(pattern_size.width-1), 1.f/(pattern_size.height-1), corners_tmp);
        
        cv::Mat corners_ref;
        vec_to_mat(corners_tmp, corners_ref);

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
    * 
    * The number of 1s in corners_status has to match the pattern size.
    * 
    * @corners_status Corner detections marked as 1 if properly detected and 0 otherwise
    * @pattern_size Size of the pattern
    */
    bool check_tracking_integrity(cv::Mat corners_status, cv::Size pattern_size)
    {
        return pattern_size.width * pattern_size.height == cv::sum(corners_status)[0];
    }


}

#endif /* calibration_hpp */