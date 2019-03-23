// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <ctime>   // localtime
#include <sstream> // stringstream
#include <iomanip> // put_time
#include <string>  // string
#include <boost/format.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
#include <boost/progress.hpp>
#include <boost/algorithm/string.hpp>
#include <math.h>

#include "utils.hpp"

bool debug = true;

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace cv;

namespace 
{ 
  const size_t ERROR_IN_COMMAND_LINE = 1; 
  const size_t SUCCESS = 0; 
  const size_t ERROR_UNHANDLED_EXCEPTION = 2; 
  const size_t FORCED_EXIT = 3;
 
} // namespace

template<typename T>
void vector_to_map(std::vector<T> v, std::map<T, int> & m)
{
    m.clear();

    for (int i = 0; i < v.size(); i++)
    {
        m[v[i]] = i;
    }
}

template <typename T>
float distancePointLine(const cv::Point_<T> point, const cv::Vec<T,3>& line)
{
  //Line is given as a*x + b*y + c = 0
  return std::fabs(line(0)*point.x + line(1)*point.y + line(2))
      / std::sqrt(line(0)*line(0)+line(1)*line(1));
}
/**
 * \brief Compute and draw the epipolar lines in two images
 *      associated to each other by a fundamental matrix
 *
 * \param title     Title of the window to display
 * \param F         Fundamental matrix
 * \param img1      First image
 * \param img2      Second image
 * \param points1   Set of points in the first image
 * \param points2   Set of points in the second image matching to the first set
 * \param inlierDistance      Points with a high distance to the epipolar lines are
 *                not displayed. If it is negative, all points are displayed
 **/
template <typename T1, typename T2>
cv::Mat drawEpipolarLines(const cv::Matx<T1,3,3> F,
                const cv::Mat& img1, const cv::Mat& img2,
                const std::vector<cv::Point_<T2>> points1,
                const std::vector<cv::Point_<T2>> points2,
                const float inlierDistance = -1)
{
  CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
  cv::Mat outImg(img1.rows, img1.cols*2, CV_8UC3);
  cv::Rect rect1(0,0, img1.cols, img1.rows);
  cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);
  /*
   * Allow color drawing
   */
  if (img1.type() == CV_8U)
  {
    cv::cvtColor(img1, outImg(rect1), CV_GRAY2BGR);
    cv::cvtColor(img2, outImg(rect2), CV_GRAY2BGR);
  }
  else
  {
    img1.copyTo(outImg(rect1));
    img2.copyTo(outImg(rect2));
  }
  std::vector<cv::Vec<T2,3>> epilines1, epilines2;
  cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1
  cv::computeCorrespondEpilines(points2, 2, F, epilines2);
 
  CV_Assert(points1.size() == points2.size() &&
        points2.size() == epilines1.size() &&
        epilines1.size() == epilines2.size());
 
  cv::RNG rng(0);
  for(size_t i=0; i<points1.size(); i++)
  {
    if(inlierDistance > 0)
    {
      if(distancePointLine(points1[i], epilines2[i]) > inlierDistance ||
        distancePointLine(points2[i], epilines1[i]) > inlierDistance)
      {
        //The point match is no inlier
        continue;
      }
    }
    /*
     * Epipolar lines of the 1st point set are drawn in the 2nd image and vice-versa
     */
    cv::Scalar color(rng(256),rng(256),rng(256));
 
    cv::line(outImg(rect2),
      cv::Point(0,-epilines1[i][2]/epilines1[i][1]),
      cv::Point(img1.cols,-(epilines1[i][2]+epilines1[i][0]*img1.cols)/epilines1[i][1]),
      color);
    cv::circle(outImg(rect1), points1[i], 3, color, -1, CV_AA);
 
    cv::line(outImg(rect1),
      cv::Point(0,-epilines2[i][2]/epilines2[i][1]),
      cv::Point(img2.cols,-(epilines2[i][2]+epilines2[i][0]*img2.cols)/epilines2[i][1]),
      color);
    cv::circle(outImg(rect2), points2[i], 3, color, -1, CV_AA);
  }
  
  return outImg;
}

template <typename FLOAT>
void computeTiltProjectionMatrix(FLOAT tauX,
    FLOAT tauY,
    Matx<FLOAT, 3, 3>* matTilt = 0,
    Matx<FLOAT, 3, 3>* dMatTiltdTauX = 0,
    Matx<FLOAT, 3, 3>* dMatTiltdTauY = 0,
    Matx<FLOAT, 3, 3>* invMatTilt = 0)
{
    FLOAT cTauX = cos(tauX);
    FLOAT sTauX = sin(tauX);
    FLOAT cTauY = cos(tauY);
    FLOAT sTauY = sin(tauY);
    Matx<FLOAT, 3, 3> matRotX = Matx<FLOAT, 3, 3>(1,0,0,0,cTauX,sTauX,0,-sTauX,cTauX);
    Matx<FLOAT, 3, 3> matRotY = Matx<FLOAT, 3, 3>(cTauY,0,-sTauY,0,1,0,sTauY,0,cTauY);
    Matx<FLOAT, 3, 3> matRotXY = matRotY * matRotX;
    Matx<FLOAT, 3, 3> matProjZ = Matx<FLOAT, 3, 3>(matRotXY(2,2),0,-matRotXY(0,2),0,matRotXY(2,2),-matRotXY(1,2),0,0,1);
    if (matTilt)
    {
        // Matrix for trapezoidal distortion of tilted image sensor
        *matTilt = matProjZ * matRotXY;
    }
    if (dMatTiltdTauX)
    {
        // Derivative with respect to tauX
        Matx<FLOAT, 3, 3> dMatRotXYdTauX = matRotY * Matx<FLOAT, 3, 3>(0,0,0,0,-sTauX,cTauX,0,-cTauX,-sTauX);
        Matx<FLOAT, 3, 3> dMatProjZdTauX = Matx<FLOAT, 3, 3>(dMatRotXYdTauX(2,2),0,-dMatRotXYdTauX(0,2),
          0,dMatRotXYdTauX(2,2),-dMatRotXYdTauX(1,2),0,0,0);
        *dMatTiltdTauX = (matProjZ * dMatRotXYdTauX) + (dMatProjZdTauX * matRotXY);
    }
    if (dMatTiltdTauY)
    {
        // Derivative with respect to tauY
        Matx<FLOAT, 3, 3> dMatRotXYdTauY = Matx<FLOAT, 3, 3>(-sTauY,0,-cTauY,0,0,0,cTauY,0,-sTauY) * matRotX;
        Matx<FLOAT, 3, 3> dMatProjZdTauY = Matx<FLOAT, 3, 3>(dMatRotXYdTauY(2,2),0,-dMatRotXYdTauY(0,2),
          0,dMatRotXYdTauY(2,2),-dMatRotXYdTauY(1,2),0,0,0);
        *dMatTiltdTauY = (matProjZ * dMatRotXYdTauY) + (dMatProjZdTauY * matRotXY);
    }
    if (invMatTilt)
    {
        FLOAT inv = 1./matRotXY(2,2);
        Matx<FLOAT, 3, 3> invMatProjZ = Matx<FLOAT, 3, 3>(inv,0,inv*matRotXY(0,2),0,inv,inv*matRotXY(1,2),0,0,1);
        *invMatTilt = matRotXY.t()*invMatProjZ;
    }
}

void my_cvUndistortPointsInternal( const CvMat* _src, CvMat* _dst, const CvMat* _cameraMatrix,
                   const CvMat* _distCoeffs,
                   const CvMat* matR, const CvMat* matP, cv::TermCriteria criteria)
{
    CV_Assert(criteria.isValid());
    double A[3][3], RR[3][3], k[14]={0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    CvMat matA=cvMat(3, 3, CV_64F, A), _Dk;
    CvMat _RR=cvMat(3, 3, CV_64F, RR);
    cv::Matx33d invMatTilt = cv::Matx33d::eye();
    cv::Matx33d matTilt = cv::Matx33d::eye();

    CV_Assert( CV_IS_MAT(_src) && CV_IS_MAT(_dst) &&
        (_src->rows == 1 || _src->cols == 1) &&
        (_dst->rows == 1 || _dst->cols == 1) &&
        _src->cols + _src->rows - 1 == _dst->rows + _dst->cols - 1 &&
        (CV_MAT_TYPE(_src->type) == CV_32FC2 || CV_MAT_TYPE(_src->type) == CV_64FC2) &&
        (CV_MAT_TYPE(_dst->type) == CV_32FC2 || CV_MAT_TYPE(_dst->type) == CV_64FC2));

    CV_Assert( CV_IS_MAT(_cameraMatrix) &&
        _cameraMatrix->rows == 3 && _cameraMatrix->cols == 3 );

    cvConvert( _cameraMatrix, &matA );


    if( _distCoeffs )
    {
        CV_Assert( CV_IS_MAT(_distCoeffs) &&
            (_distCoeffs->rows == 1 || _distCoeffs->cols == 1) &&
            (_distCoeffs->rows*_distCoeffs->cols == 4 ||
             _distCoeffs->rows*_distCoeffs->cols == 5 ||
             _distCoeffs->rows*_distCoeffs->cols == 8 ||
             _distCoeffs->rows*_distCoeffs->cols == 12 ||
             _distCoeffs->rows*_distCoeffs->cols == 14));

        _Dk = cvMat( _distCoeffs->rows, _distCoeffs->cols,
            CV_MAKETYPE(CV_64F,CV_MAT_CN(_distCoeffs->type)), k);

        cvConvert( _distCoeffs, &_Dk );
        if (k[12] != 0 || k[13] != 0)
        {
            computeTiltProjectionMatrix<double>(k[12], k[13], NULL, NULL, NULL, &invMatTilt);
            computeTiltProjectionMatrix<double>(k[12], k[13], &matTilt, NULL, NULL);
        }
    }

    if( matR )
    {
        CV_Assert( CV_IS_MAT(matR) && matR->rows == 3 && matR->cols == 3 );
        cvConvert( matR, &_RR );
    }
    else
        cvSetIdentity(&_RR);

    if( matP )
    {
        double PP[3][3];
        CvMat _P3x3, _PP=cvMat(3, 3, CV_64F, PP);
        CV_Assert( CV_IS_MAT(matP) && matP->rows == 3 && (matP->cols == 3 || matP->cols == 4));
        cvConvert( cvGetCols(matP, &_P3x3, 0, 3), &_PP );
        cvMatMul( &_PP, &_RR, &_RR );
    }

    const CvPoint2D32f* srcf = (const CvPoint2D32f*)_src->data.ptr;
    const CvPoint2D64f* srcd = (const CvPoint2D64f*)_src->data.ptr;
    CvPoint2D32f* dstf = (CvPoint2D32f*)_dst->data.ptr;
    CvPoint2D64f* dstd = (CvPoint2D64f*)_dst->data.ptr;
    int stype = CV_MAT_TYPE(_src->type);
    int dtype = CV_MAT_TYPE(_dst->type);
    int sstep = _src->rows == 1 ? 1 : _src->step/CV_ELEM_SIZE(stype);
    int dstep = _dst->rows == 1 ? 1 : _dst->step/CV_ELEM_SIZE(dtype);

    double fx = A[0][0];
    double fy = A[1][1];
    double ifx = 1./fx;
    double ify = 1./fy;
    double cx = A[0][2];
    double cy = A[1][2];

    int n = _src->rows + _src->cols - 1;
    for( int i = 0; i < n; i++ )
    {
        double x, y, x0 = 0, y0 = 0, u, v;
        if( stype == CV_32FC2 )
        {
            x = srcf[i*sstep].x;
            y = srcf[i*sstep].y;
        }
        else
        {
            x = srcd[i*sstep].x;
            y = srcd[i*sstep].y;
        }
        u = x; v = y;
        x = (x - cx)*ifx;
        y = (y - cy)*ify;

        if( _distCoeffs ) {
            // compensate tilt distortion
            cv::Vec3d vecUntilt = invMatTilt * cv::Vec3d(x, y, 1);
            double invProj = vecUntilt(2) ? 1./vecUntilt(2) : 1;
            x0 = x = invProj * vecUntilt(0);
            y0 = y = invProj * vecUntilt(1);

            double error = std::numeric_limits<double>::max();
            // compensate distortion iteratively

            for( int j = 0; ; j++ )
            {
                if ((criteria.type & cv::TermCriteria::COUNT) && j >= criteria.maxCount)
                    break;
                if ((criteria.type & cv::TermCriteria::EPS) && error < criteria.epsilon)
                    break;
                double r2 = x*x + y*y;
                double icdist = (1 + ((k[7]*r2 + k[6])*r2 + k[5])*r2)/(1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2);
                double deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x)+ k[8]*r2+k[9]*r2*r2;
                double deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y+ k[10]*r2+k[11]*r2*r2;
                x = (x0 - deltaX)*icdist;
                y = (y0 - deltaY)*icdist;

                if(criteria.type & cv::TermCriteria::EPS)
                {
                    double r4, r6, a1, a2, a3, cdist, icdist2;
                    double xd, yd, xd0, yd0;
                    cv::Vec3d vecTilt;

                    r2 = x*x + y*y;
                    r4 = r2*r2;
                    r6 = r4*r2;
                    a1 = 2*x*y;
                    a2 = r2 + 2*x*x;
                    a3 = r2 + 2*y*y;
                    cdist = 1 + k[0]*r2 + k[1]*r4 + k[4]*r6;
                    icdist2 = 1./(1 + k[5]*r2 + k[6]*r4 + k[7]*r6);
                    xd0 = x*cdist*icdist2 + k[2]*a1 + k[3]*a2 + k[8]*r2+k[9]*r4;
                    yd0 = y*cdist*icdist2 + k[2]*a3 + k[3]*a1 + k[10]*r2+k[11]*r4;

                    vecTilt = matTilt*cv::Vec3d(xd0, yd0, 1);
                    invProj = vecTilt(2) ? 1./vecTilt(2) : 1;
                    xd = invProj * vecTilt(0);
                    yd = invProj * vecTilt(1);

                    double x_proj = xd*fx + cx;
                    double y_proj = yd*fy + cy;

                    error = sqrt( pow(x_proj - u, 2) + pow(y_proj - v, 2) );
                }
            }
        } 

        double xx = RR[0][0]*x + RR[0][1]*y + RR[0][2];
        double yy = RR[1][0]*x + RR[1][1]*y + RR[1][2];
        double ww = 1./(RR[2][0]*x + RR[2][1]*y + RR[2][2]);
        x = xx*ww;
        y = yy*ww;

        if( dtype == CV_32FC2 )
        {
            dstf[i*dstep].x = (float)x;
            dstf[i*dstep].y = (float)y;
        }
        else
        {
            dstd[i*dstep].x = x;
            dstd[i*dstep].y = y;
        }
    }
}

void my_undistortPoints(InputArray _src, OutputArray _dst,
                     InputArray _cameraMatrix,
                     InputArray _distCoeffs,
                     InputArray _Rmat,
                     InputArray _Pmat,
                     TermCriteria criteria)
{
    Mat src = _src.getMat(), cameraMatrix = _cameraMatrix.getMat();
    Mat distCoeffs = _distCoeffs.getMat(), R = _Rmat.getMat(), P = _Pmat.getMat();

    CV_Assert( src.isContinuous() && (src.depth() == CV_32F || src.depth() == CV_64F) &&
              ((src.rows == 1 && src.channels() == 2) || src.cols*src.channels() == 2));

    _dst.create(src.size(), src.type(), -1, true);
    Mat dst = _dst.getMat();

    CvMat _csrc = cvMat(src), _cdst = cvMat(dst), _ccameraMatrix = cvMat(cameraMatrix);
    CvMat matR, matP, _cdistCoeffs, *pR=0, *pP=0, *pD=0;
    if( !R.empty() )
        pR = &(matR = cvMat(R));
    if( !P.empty() )
        pP = &(matP = cvMat(P));
    if( !distCoeffs.empty() )
        pD = &(_cdistCoeffs = cvMat(distCoeffs));
    my_cvUndistortPointsInternal(&_csrc, &_cdst, &_ccameraMatrix, pD, pR, pP, criteria);
}

void _undistortPoints(InputArray _src, OutputArray _dst,
                     InputArray _cameraMatrix,
                     InputArray _distCoeffs,
                     InputArray _Rmat,
                     InputArray _Pmat = noArray())
{
    my_undistortPoints(_src, _dst, _cameraMatrix, _distCoeffs, _Rmat, _Pmat, TermCriteria(TermCriteria::MAX_ITER, 5, 0.01));
}



int main(int argc, char * argv[]) try
{
    /* -------------------------------- */
    /*   Command line argument parsing  */
    /* -------------------------------- */
    
    std::string corners_file_1, corners_file_2, intrinsics_file_1, intrinsics_file_2;
    bool vflip = false;
    bool verbose = false;

    po::options_description desc("Program options");
    desc.add_options()
        ("help,h", "Print help messages")
        // ("corners,c", po::value<std::string>()->default_value("./corners.yml"), "")
        // ("corner-selection,s", po::value<std::string>()->default_value("./corner-selection.yml"), "")
        // ("intrinsics,i", po::value<std::string>()->default_value("./intrinsics.yml"), "")
        // ("modality,m", po::value<std::string>()->default_value("thermal"), "Visual modality")
        // ("file-ext,x", po::value<std::string>()->default_value(".jpg"), "Image file extension")
        // ("verbose,v", po::bool_switch(&verbose), "Verbosity")
        ("nb-clusters,k", po::value<int>()->default_value(50), "Number of k-means clusters")
        ("vflip,f", po::bool_switch(&vflip), "Vertical flip registered images")
        ("extrinsics-file,e", po::value<std::string>()->default_value(""), "Extrinsics")
        ("output-parameters,o", po::value<std::string>()->default_value(""), "Output parameters")
        ("corners-1", po::value<std::string>(&corners_file_1)->required(), "")
        ("corners-2", po::value<std::string>(&corners_file_2)->required(), "")
        ("intrinsics-1", po::value<std::string>(&intrinsics_file_1)->required(), "")
        ("intrinsics-2", po::value<std::string>(&intrinsics_file_2)->required(), "");
    
    po::positional_options_description positional_options; 
    positional_options.add("corners-1", 1); 
    positional_options.add("corners-2", 1); 
    positional_options.add("intrinsics-1", 1); 
    positional_options.add("intrinsics-2", 1); 

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm); // can throw

    // --help option?
    if (vm.count("help"))
    {
        std::cout << "Basic Command Line Parameter App" << std::endl
        << desc << std::endl;
        return SUCCESS;
    }
    
    po::notify(vm); // throws on error, so do after help in case
    
    /* --------------- */
    /*    Main code    */
    /* --------------- */

    cv::FileStorage corners_fs_1 (corners_file_1, cv::FileStorage::READ);
    cv::FileStorage corners_fs_2 (corners_file_2, cv::FileStorage::READ);

    std::string prefix_1, prefix_2;
    corners_fs_1["prefix"] >> prefix_1;
    corners_fs_2["prefix"] >> prefix_2;

    std::string file_ext_1, file_ext_2;
    corners_fs_1["file-extension"] >> file_ext_1;
    corners_fs_2["file-extension"] >> file_ext_2;

    std::string modality_1, modality_2;
    corners_fs_1["modality"] >> modality_1;
    corners_fs_2["modality"] >> modality_2;

    cv::Size frame_size_1, frame_size_2;
    corners_fs_1["resize_dims"] >> frame_size_1;
    corners_fs_2["resize_dims"] >> frame_size_2;

    int nb_sequence_dirs_1, nb_sequence_dirs_2;
    corners_fs_1["sequence_dirs"] >> nb_sequence_dirs_1;
    corners_fs_2["sequence_dirs"] >> nb_sequence_dirs_2;
    assert(nb_sequence_dirs_1 == nb_sequence_dirs_2);

    cv::Size pattern_size_1, pattern_size_2;
    corners_fs_1["pattern_size"] >> pattern_size_1;
    corners_fs_2["pattern_size"] >> pattern_size_2;

    int y_shift_1, y_shift_2;
    corners_fs_1["y-shift"] >> y_shift_1;
    corners_fs_2["y-shift"] >> y_shift_2;

    std::vector<std::string> frames_all_1, frames_all_2;
    cv::Mat corners_all_1, corners_all_2;

    std::vector<std::vector<std::pair<int,int> > > frames_indices_all;
    for (int i = 0; i < nb_sequence_dirs_1; i++)
    {
        std::string sequence_dir_1, sequence_dir_2;
        corners_fs_1["sequence_dir-" + std::to_string(i)] >> sequence_dir_1;
        corners_fs_2["sequence_dir-" + std::to_string(i)] >> sequence_dir_2;
        assert(sequence_dir_1 == sequence_dir_2);

        std::vector<uls::Timestamp> log_1 = uls::read_log_file(fs::path(sequence_dir_1) / fs::path(corners_fs_1["log-file"]));
        std::vector<uls::Timestamp> log_2 = uls::read_log_file(fs::path(sequence_dir_2) / fs::path(corners_fs_2["log-file"]));
        std::vector<std::pair<uls::Timestamp,uls::Timestamp> > log_12;
        uls::time_sync(log_1, log_2, log_12);

        std::vector<std::string> frames_1, frames_2;
        corners_fs_1["frames-" + std::to_string(i)] >> frames_1;
        corners_fs_2["frames-" + std::to_string(i)] >> frames_2;

        std::map<std::string, int> map_1, map_2;
        vector_to_map<std::string>(frames_1, map_1);
        vector_to_map<std::string>(frames_2, map_2);

        cv::Mat corners_1, corners_2;
        corners_fs_1["corners-" + std::to_string(i)] >> corners_1;
        corners_fs_2["corners-" + std::to_string(i)] >> corners_2;

        std::vector<std::string> frames_1_aux, frames_2_aux;
        std::vector<cv::Mat> corners_1_aux, corners_2_aux;

        std::map<std::string, int>::iterator it_1, it_2;
        for (int j = 0; j < log_12.size(); j++)
        {
            std::string frame_path_1 = prefix_1 + log_12[j].first.id  + file_ext_1;
            std::string frame_path_2 = prefix_2 + log_12[j].second.id + file_ext_2;

            it_1 = map_1.find(frame_path_1);
            it_2 = map_2.find(frame_path_2);
            if (it_1 != map_1.end() && it_2 != map_2.end())
            {
                frames_all_1.push_back((fs::path(sequence_dir_1) / fs::path(frames_1[it_1->second])).string());
                frames_all_2.push_back((fs::path(sequence_dir_2) / fs::path(frames_2[it_2->second])).string());
                // Re-orient corners so first corner is the top-left and last corner the bottom-right one
                corners_all_1.push_back( uls::orient_corners(corners_1.row(it_1->second)) );
                corners_all_2.push_back( uls::orient_corners(corners_2.row(it_2->second)) );
            }
        }
    }

    corners_fs_1.release();
    corners_fs_2.release();

    assert(frames_all_1.size() == frames_all_2.size());
    assert(corners_all_1.rows == corners_all_2.rows);

    int K = vm["nb-clusters"].as<int>();

    cv::Mat labels, centers;
    cv::kmeans(corners_all_1, K, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

    // ---

    std::vector<std::vector<int> > indices;
    for (int k = 0; k < K; k++)
    {
        std::vector<int> indices_k;
        for (int i = 0; i < labels.rows; i++)
            if (labels.at<int>(i,0) == k) indices_k.push_back(i);

        auto rng = std::default_random_engine {};
        std::shuffle(indices_k.begin(), indices_k.end(), rng);
        indices.push_back(indices_k);
    }
    
    std::vector<cv::Mat> corners_tmp_1, corners_tmp_2;
    corners_tmp_1.resize(K);
    corners_tmp_2.resize(K);

    std::vector<std::string> corner_frames_tmp_1, corner_frames_tmp_2;
    corner_frames_tmp_1.resize(K);
    corner_frames_tmp_2.resize(K);

    cv::Size pattern_size;

    int k = 0;
    std::vector<int> ptr (K);
    bool keep_selecting = true;

    // while (keep_selecting)
    // {
    //     std::cout << k << ":" << ptr[k]+1 << "/" << indices[k].size() << std::endl;
    //     int idx = indices[k][ptr[k]];//indices_k.at<int>(i,0);

    //     cv::Mat corners_row_1_transf, corners_row_2_transf;
    //     cv::Size frame_size_transf;
    //     uls::transform_point_domains(corners_all_1.row(idx), corners_all_2.row(idx), 
    //                                  frame_size_1, frame_size_2, 
    //                                  corners_row_1_transf, corners_row_2_transf, frame_size_transf);

    //     cv::Mat img_1;
    //     if (modality_1 == "Color")
    //         img_1 = uls::ColorFrame(fs::path(frames_all_1[idx]), frame_size_transf, y_shift_1).mat();
    //     else if (modality_1 == "Thermal")
    //         img_1 = uls::ThermalFrame(fs::path(frames_all_1[idx]), frame_size_transf, y_shift_2).mat();

    //     cv::Mat img_2;
    //     if (modality_2 == "Color")
    //         img_2 = uls::ColorFrame(fs::path(frames_all_2[idx]), frame_size_transf, y_shift_1).mat();
    //     else if (modality_2 == "Thermal")
    //         img_2 = uls::ThermalFrame(fs::path(frames_all_2[idx]), frame_size_transf, y_shift_2).mat();

    //     cv::cvtColor(img_1, img_1, cv::COLOR_GRAY2BGR);
    //     cv::cvtColor(img_2, img_2, cv::COLOR_GRAY2BGR);

    //     cv::Mat corners_row_1_aligned, corners_row_2_aligned;
    //     // uls::align_pattern_corners(corners_all_1.row(idx), corners_all_2.row(idx), pattern_size_1, pattern_size_2, corners_row_1, corners_row_2, pattern_size);
    //     uls::align_pattern_corners(corners_row_1_transf, corners_row_2_transf, 
    //                                pattern_size_1, pattern_size_2, 
    //                                corners_row_1_aligned, corners_row_2_aligned, pattern_size);
    //     cv::drawChessboardCorners(img_1, pattern_size, corners_row_1_aligned, true);
    //     cv::drawChessboardCorners(img_2, pattern_size, corners_row_2_aligned, true);

    //     // cv::Mat corners_row_1minus2 = corners_row_1_aligned - corners_row_2_aligned;
    //     // std::cout << corners_row_1minus2 << std::endl;
    //     // cv::Point2f pi_1, pf_1, pi_2, pf_2;
    //     // pi_1 = corners_row_1_aligned.at<cv::Point2f>(0,0);
    //     // pi_2 = corners_row_2_aligned.at<cv::Point2f>(0,0);
    //     // pf_1 = corners_row_1_aligned.at<cv::Point2f>(0,corners_row_1_aligned.cols-1);
    //     // pf_2 = corners_row_2_aligned.at<cv::Point2f>(0,corners_row_1_aligned.cols-1);

    //     // std::cout << pi_1 << "->" << pf_1 << std::endl;
    //     // std::cout << pi_2 << "->" << pf_2 << std::endl;

    //     std::vector<cv::Mat> tiling = {img_1, img_2};
    //     cv::Mat img;
    //     uls::tile(tiling, 800, 900, 1, 2, img);

    //     std::stringstream ss;
    //     if (corner_frames_tmp_1[k] == frames_all_1[idx])
    //         ss << "[*" << k << "*]";
    //     else
    //         ss << "[ " << k << " ]";
    //     ss << ' ' << ptr[k] << '/' << indices[k].size(); 
    //     cv::putText(img, ss.str(), cv::Point(img.cols/20.0,img.rows/20.0), CV_FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0,0,255), 1, 8, false);
    //     cv::imshow("Viewer", img);

    //     char ret = cv::waitKey();
    //     if (ret == 'j')
    //         ptr[k] = (ptr[k] > 0) ? ptr[k] - 1 : ptr[k];
    //     else if (ret == ';')
    //         ptr[k] = (ptr[k] < (indices[k].size() - 1)) ? ptr[k] + 1 : ptr[k];
    //     else if (ret == 'k')
    //     {
    //         k = (k > 0) ? k - 1 : K-1;
    //     }
    //     else if (ret == 'l' || ret == ' ')
    //         k = (k < (K - 1)) ? k + 1 : 0;
    //     else if (ret == 13) 
    //     {
    //         if (corner_frames_tmp_1[k] == frames_all_1[idx])
    //         {
    //             corner_frames_tmp_1[k] = corner_frames_tmp_2[k] = std::string();
    //             corners_tmp_1[k] = corners_tmp_2[k] = cv::Mat();
    //         }
    //         else
    //         {
    //             corners_tmp_1[k] = corners_row_1_aligned;// corners_all_1.row(idx);
    //             corners_tmp_2[k] = corners_row_2_aligned;//corners_all_2.row(idx);
    //             corner_frames_tmp_1[k] = frames_all_1[idx];
    //             corner_frames_tmp_2[k] = frames_all_2[idx];
    //             k = (k < (K - 1)) ? k + 1 : 0;
    //         }
    //     }
    //     else if (ret == 27)
    //         keep_selecting = false;
    // }

    // assert(corner_frames_tmp_1.size() == corner_frames_tmp_2.size());

    cv::Mat corners_selection_1, corners_selection_2;
    std::vector<std::string> frames_selection_1, frames_selection_2;
    
    // for (int k = 0; k < corner_frames_tmp_1.size(); k++)
    // {
    //     if (!corner_frames_tmp_1[k].empty())
    //     {
    //         corners_selection_1.push_back(corners_tmp_1[k]);
    //         corners_selection_2.push_back(corners_tmp_2[k]);
    //         frames_selection_1.push_back(corner_frames_tmp_1[k]);
    //         frames_selection_2.push_back(corner_frames_tmp_2[k]);
    //     }
    // }

    cv::FileStorage extrinsics_fs;

    // if (!vm["extrinsics-file"].as<std::string>().empty())
    // {
    //     extrinsics_fs.open(vm["extrinsics-file"].as<std::string>(), cv::FileStorage::WRITE);
    //     extrinsics_fs << "corners_selection_1" << corners_selection_1;
    //     extrinsics_fs << "corners_selection_2" << corners_selection_2;
    //     extrinsics_fs << "frames_selection_1" << frames_selection_1;
    //     extrinsics_fs << "frames_selection_2" << frames_selection_2;
    //     extrinsics_fs.release();
    // }

    if (!vm["extrinsics-file"].as<std::string>().empty())
    {
        extrinsics_fs.open(vm["extrinsics-file"].as<std::string>(), cv::FileStorage::READ);
        extrinsics_fs["corners_selection_1"] >> corners_selection_1;
        extrinsics_fs["corners_selection_2"] >> corners_selection_2;
        extrinsics_fs["frames_selection_1"] >> frames_selection_1;
        extrinsics_fs["frames_selection_2"] >> frames_selection_2;
        extrinsics_fs.release();
    }

    cv::FileStorage intrinsics_fs_1 (intrinsics_file_1, cv::FileStorage::READ);
    cv::FileStorage intrinsics_fs_2 (intrinsics_file_2, cv::FileStorage::READ);

    cv::Mat camera_matrix_1, camera_matrix_2;
    cv::Mat dist_coeffs_1, dist_coeffs_2;
    intrinsics_fs_1["camera_matrix"] >> camera_matrix_1;
    intrinsics_fs_2["camera_matrix"] >> camera_matrix_2;
    intrinsics_fs_1["dist_coeffs"] >> dist_coeffs_1;
    intrinsics_fs_2["dist_coeffs"] >> dist_coeffs_2;

    intrinsics_fs_1.release();
    intrinsics_fs_2.release();

    std::vector<std::vector<cv::Point2f> > image_points_1, image_points_2;
    uls::mat_to_vecvec<cv::Point2f>(corners_selection_1, image_points_1);
    uls::mat_to_vecvec<cv::Point2f>(corners_selection_2, image_points_2);

    std::vector<std::vector<cv::Point3f> > object_points (1);
    uls::calcBoardCornerPositions(cv::Size(9,6), 0.05f, 0.05f, object_points[0]);
    object_points.resize(image_points_1.size(), object_points[0]);

    cv::Mat R, T, E, F;
    int flags = CV_CALIB_USE_INTRINSIC_GUESS + 
                CV_CALIB_FIX_ASPECT_RATIO + 
                CV_CALIB_FIX_FOCAL_LENGTH + 
                CV_CALIB_FIX_K1 + 
                CV_CALIB_FIX_K2 + 
                CV_CALIB_FIX_K3 + 
                CV_CALIB_FIX_K4 +
                CV_CALIB_FIX_K5 + 
                CV_CALIB_FIX_K6;// + CV_CALIB_TILTED_MODEL;
    double rms = cv::stereoCalibrate(object_points,
                                     image_points_1, image_points_2, 
                                     camera_matrix_1, dist_coeffs_1, 
                                     camera_matrix_2, dist_coeffs_2,
                                     cv::Size(1280, 720+abs(y_shift_1)),
                                     R, T, E, F,
                                     flags,
                                     cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 100, 1e-2));
    std::cout << rms << std::endl;

    T.at<double>(1,0) = 0;

    cv::Mat R1,R2,P1,P2,Q;
    cv::stereoRectify(camera_matrix_1, dist_coeffs_1, camera_matrix_2, dist_coeffs_2, cv::Size(1280,720+abs(y_shift_1)), R, T, R1, R2, P1, P2, Q, 0);//, cv::Size(1.2*1280,1.2*(720+abs(y_shift_1))), &r1, &r2);
    cv::Mat R1z,R2z,P1z,P2z,Qz;
    cv::stereoRectify(camera_matrix_1, dist_coeffs_1, camera_matrix_2, dist_coeffs_2, cv::Size(1280,720+abs(y_shift_2)), R, T, R1z, R2z, P1z, P2z, Qz, cv::CALIB_ZERO_DISPARITY);

    cv::Mat mapi1_1, mapi2_1, mapi1_2, mapi2_2;
    cv::initUndistortRectifyMap(camera_matrix_1, dist_coeffs_1, cv::Mat(), cv::Mat(), cv::Size(1280,720+abs(y_shift_1)), CV_32FC1, mapi1_1, mapi2_1);
    cv::initUndistortRectifyMap(camera_matrix_2, dist_coeffs_2, cv::Mat(), cv::Mat(), cv::Size(1280,720+abs(y_shift_2)), CV_32FC1, mapi1_2, mapi2_2);
    cv::Mat mape1_1, mape2_1, mape1_2, mape2_2;
    cv::initUndistortRectifyMap(camera_matrix_1, dist_coeffs_1, R1, P1, cv::Size(1280,720+abs(y_shift_1)), CV_32FC1, mape1_1, mape2_1);
    cv::initUndistortRectifyMap(camera_matrix_2, dist_coeffs_2, R2, P2, cv::Size(1280,720+abs(y_shift_2)), CV_32FC1, mape1_2, mape2_2);
    cv::Mat mapez1_1, mapez2_1, mapez1_2, mapez2_2;
    cv::initUndistortRectifyMap(camera_matrix_1, dist_coeffs_1, R1z, P1z, cv::Size(1280,720+abs(y_shift_1)), CV_32FC1, mapez1_1, mapez2_1);
    cv::initUndistortRectifyMap(camera_matrix_2, dist_coeffs_2, R2z, P2z, cv::Size(1280,720+abs(y_shift_2)), CV_32FC1, mapez1_2, mapez2_2);
    
    if (!vm["output-parameters"].as<std::string>().empty())
    {
        extrinsics_fs.open(vm["output-parameters"].as<std::string>(), cv::FileStorage::WRITE);
        extrinsics_fs << "modality-1" << modality_1;
        extrinsics_fs << "modality-2" << modality_2;
        extrinsics_fs << "mapx-1" << mape1_1;
        extrinsics_fs << "mapy-1" << mape2_1;
        extrinsics_fs << "mapx-2" << mape1_2;
        extrinsics_fs << "mapy-2" << mape2_2;
        extrinsics_fs << "y-shift-1" << y_shift_1;
        extrinsics_fs << "y-shift-2" << y_shift_2;
        extrinsics_fs << "vflip" << vflip;

        extrinsics_fs.release();
    }

    int grid_x = 3;
    int grid_y = 2;
    while (true)
    {
        for (int i = 0; i < frames_all_1.size(); i++)
        {
            cv::Mat img_1 = uls::ColorFrame(fs::path(frames_all_1[i]), cv::Size(1280,720), y_shift_1).mat();
            cv::Mat img_2 = uls::ThermalFrame(fs::path(frames_all_2[i]), cv::Size(1280,720), y_shift_2).mat();
        
            std::vector<cv::Mat> tiling (grid_x * grid_y);
            cv::cvtColor(img_1, tiling[0], cv::COLOR_GRAY2BGR);
            cv::cvtColor(img_2, tiling[1], cv::COLOR_GRAY2BGR);
            cv::multiply(tiling[0], cv::Scalar(0,0,1), tiling[0]);
            cv::multiply(tiling[1], cv::Scalar(0,1,0), tiling[1]);
            cv::addWeighted(tiling[0], 1, tiling[1], 1, 0.0, tiling[2]);
        
            cv::Mat tmp_1, tmp_2, tmp_r;

            cv::remap(img_1, tmp_1, mapi1_1, mapi2_1, cv::INTER_LINEAR);
            cv::remap(img_2, tmp_2, mapi1_2, mapi2_2, cv::INTER_LINEAR);
            cv::cvtColor(tmp_1, tmp_1, cv::COLOR_GRAY2BGR);
            cv::cvtColor(tmp_2, tmp_2, cv::COLOR_GRAY2BGR);
            cv::multiply(tmp_1, cv::Scalar(0,0,1), tmp_1);
            cv::multiply(tmp_2, cv::Scalar(0,1,0), tmp_2);
            cv::addWeighted(tmp_1, 1, tmp_2, 1, 0.0, tiling[3]);

            cv::remap(img_1, tmp_1, mape1_1, mape2_1, cv::INTER_LINEAR);
            cv::remap(img_2, tmp_2, mape1_2, mape2_2, cv::INTER_LINEAR);
            cv::cvtColor(tmp_1, tmp_1, cv::COLOR_GRAY2BGR);
            cv::cvtColor(tmp_2, tmp_2, cv::COLOR_GRAY2BGR);
            cv::multiply(tmp_1, cv::Scalar(0,0,1), tmp_1);
            cv::multiply(tmp_2, cv::Scalar(0,1,0), tmp_2);
            cv::addWeighted(tmp_1, 1, tmp_2, 1, 0.0, tmp_r);
            if (!vflip) tiling[4] = tmp_r;
            else cv::flip(tmp_r, tiling[4], 0);

            cv::remap(img_1, tmp_1, mapez1_1, mapez2_1, cv::INTER_LINEAR);
            cv::remap(img_2, tmp_2, mapez1_2, mapez2_2, cv::INTER_LINEAR);
            cv::cvtColor(tmp_1, tmp_1, cv::COLOR_GRAY2BGR);
            cv::cvtColor(tmp_2, tmp_2, cv::COLOR_GRAY2BGR);
            cv::multiply(tmp_1, cv::Scalar(0,0,1), tmp_1);
            cv::multiply(tmp_2, cv::Scalar(0,1,0), tmp_2);
            cv::addWeighted(tmp_1, 1, tmp_2, 1, 0.0, tmp_r);
            if (!vflip) tiling[5] = tmp_r;
            else cv::flip(tmp_r, tiling[5], 0);

            cv::Mat viewer_img;
            uls::tile(tiling, 1920, 640, grid_x, grid_y, viewer_img);
            cv::imshow("Viewer", viewer_img);
            cv::waitKey(33); 
        }
    }

    return SUCCESS;
}
catch(po::error& e)
{
    std::cerr << "Error parsing arguments: " << e.what() << std::endl;
    return ERROR_IN_COMMAND_LINE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return ERROR_UNHANDLED_EXCEPTION;
}

