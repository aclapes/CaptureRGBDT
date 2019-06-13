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
#include <boost/iterator.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>
#include <boost/progress.hpp>
#include <boost/algorithm/string.hpp>

#include "utils/common.hpp"
#include "utils/calibration.hpp"

bool debug = true;

namespace po = boost::program_options;
namespace fs = boost::filesystem;

namespace 
{ 
  const size_t ERROR_IN_COMMAND_LINE = 1; 
  const size_t SUCCESS = 0; 
  const size_t ERROR_UNHANDLED_EXCEPTION = 2; 
  const size_t FORCED_EXIT = 3;
 
} // namespace

// template<typename T>
void find_chessboard_corners(std::vector<std::string> frames, 
                             cv::Size pattern_size, 
                             std::vector<cv::Mat> & frames_corners,
                             std::vector<int> & frames_inds,
                             cv::Size resize_dims = cv::Size(),
                             std::string prefix = "",
                            //  int y_shift = 0,
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
        // cv::Mat img = T(fs::path(prefix) / fs::path(frames[i]), resize_dims).get();
        cv::Mat img = cv::imread((fs::path(prefix) / fs::path(frames[i])).string(), cv::IMREAD_UNCHANGED);
        uls::resize(img, img, resize_dims);

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
            if ( ! uls::check_corners_integrity(status, pattern_size) )
            {
                tracking_enabled = false;
                corners.release();
            }
        }

        if (tracking_enabled)
        {
            if ((corners.rows == pattern_size.width * pattern_size.height) && uls::check_corners_2d_positions(corners, pattern_size))
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


int main(int argc, char * argv[]) try
{
    /* -------------------------------- */
    /*   Command line argument parsing  */
    /* -------------------------------- */
    
    std::string input_list_file_str;
    std::string modality_str;
    std::string output_file_str;
    int verbose;

    po::options_description desc("Program options");
    desc.add_options()
        ("help,h", "Print help messages")
        // ("corners,c", po::value<std::string>()->default_value("./corners.yml"), "")
        // ("corner-selection,s", po::value<std::string>()->default_value("./corner-selection.yml"), "")
        // ("intrinsics,i", po::value<std::string>()->default_value("./intrinsics.yml"), "")
        // ("modality,m", po::value<std::string>()->default_value("Thermal"), "Visual modality")
        ("input-file-or-dir", po::value<std::string>(&input_list_file_str)->required(), "File containing list of calibration sequence directories")
        ("prefix,f", po::value<std::string>()->default_value(""), "Prefix")
        // ("log-file,l", po::value<std::string>()->default_value(""), "Log file (e.g. rs.log)")
        // ("file-ext,x", po::value<std::string>()->default_value(".jpg"), "Image file extension")
        ("pattern-size,p", po::value<std::string>()->default_value("11,8"), "Pattern size \"x,y\" squares")
        ("resize-dims,r", po::value<std::string>()->default_value("960,720"), "Resize frame to (h,w)")
        //  ("y-shift,y", po::value<int>()->default_value(0), "Y-shift")
        // ("verbose,v", po::bool_switch(&verbose), "Verbosity")
        ("verbose,v", po::value<int>()->default_value(0), "")
        // ("modality", po::value<std::string>(&modality_str)->required(), "Modality (either Color or Thermal)")
        ("output-file", po::value<std::string>(&output_file_str)->required(), "Output file");

    
    po::positional_options_description positional_options; 
    positional_options.add("input-file-or-dir", 1); 
    positional_options.add("prefix", 1); 
    positional_options.add("output-file", 1); 

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

    // process all program arguments ---

    // read verbosity level from program arguments (po)
    verbose = vm["verbose"].as<int>();

    // read calibration pattern size from po
    std::vector<std::string> pattern_size_aux;
    boost::split(pattern_size_aux, vm["pattern-size"].as<std::string>(), boost::is_any_of(","));
    assert(pattern_size_aux.size() == 2);

    int x = std::stoi(pattern_size_aux[0]);
    int y = std::stoi(pattern_size_aux[1]);
    assert(x > 2 && y > 2);
    cv::Size pattern_size (x,y);

    // read prefix directory for this particular modality from po, e.g. rs/color, rs/depth, or pt/thermal
    std::string prefix = vm["prefix"].as<std::string>();
    boost::trim_if(prefix, &uls::is_bar);  // remove leading and tailing '/' and '\' bars

    // read image resize dimensions from po, e.g 1280,720,
    // to resize frames to w x h before trying to find chessboard corners
    cv::Size resize_dims;
    if (vm.find("resize-dims") != vm.end())
    {
        std::vector<std::string> resize_dims_aux;
        boost::split(resize_dims_aux, vm["resize-dims"].as<std::string>(), boost::is_any_of(","));

        int w, h;
        if (resize_dims_aux.size() < 1)
            return EXIT_FAILURE;
        else if (resize_dims_aux.size() == 1)
            w = h = std::stoi(resize_dims_aux[0]);
        else
        {
            w = std::stoi(resize_dims_aux[0]);
            h = std::stoi(resize_dims_aux[1]);
        }
        resize_dims = cv::Size(w,h); 
    }
    
    // --- end process program arguments
    
    // list the calibration sequences within input-file-or-dir, which can be either
    // a file listing diretories or a directory of directories
    std::vector<std::string> sequence_dirs;

    fs::path input_p (input_list_file_str);
    if (fs::is_directory(input_p))  // if a directory
    {
        for(auto& entry : boost::make_iterator_range(fs::directory_iterator(input_p), {}))
            sequence_dirs.push_back(entry.path().string());
    }
    else  // if a file
    {
        std::ifstream dir_list_reader;
        dir_list_reader.open(input_list_file_str);
        if (dir_list_reader.is_open())
        {
            std::string line;
            while (std::getline(dir_list_reader, line))
                sequence_dirs.push_back(line);
            dir_list_reader.close();
        }
    }
    
    if (sequence_dirs.empty())
    {
        std::cerr << "Calibration file (input-file-or-dir argument) not found." << std::endl;
        return EXIT_FAILURE;
    }
    
    // list all the frames in the sequences as a hash map: (sequence_name, list_of_sequence_frame_filepaths)

    std::map<std::string, std::vector<std::string> > sequences;
    for (std::string seq_dir : sequence_dirs)
    {
        std::vector<std::string> s = uls::list_images_in_directory(seq_dir, vm["prefix"].as<std::string>());  // lists jpg/png files
        std::sort(s.begin(), s.end());  // sort alphabetically
        sequences[seq_dir] = s;
    }

    cv::FileStorage fstorage (vm["output-file"].as<std::string>(), cv::FileStorage::WRITE);
    fstorage << "prefix" << prefix;
    fstorage << "pattern_size" << pattern_size;
    fstorage << "resize_dims" << resize_dims;
    // fstorage << "modality" << vm["prefix"].as<std::string>(); //modality_str;
    // fstorage << "y-shift" << vm["y-shift"].as<int>();
    // fstorage << "log-file" << vm["log-file"].as<std::string>();
    // fstorage << "file-extension" << vm["file-ext"].as<std::string>();
    fstorage << "nb_sequences" << ((int) sequence_dirs.size());

    // // std::vector<int> fids; // keep track
    // // boost::progress_display pd (sequences.size());
    std::map<std::string, std::vector<std::string> >::iterator it;
    int i;
    std::string serial_number;

    for (it = sequences.begin(), i = 0; it != sequences.end(); it++, i++)
    {
        // Consistency check: all calibration sequences were captures with the same camera
        // ---
        cv::FileStorage fs ( (fs::path(it->first) / fs::path("rs_info.yml")).string(), cv::FileStorage::READ );
        std::string sn;
        if (fs.isOpened())
            fs["serial_number"] >> sn;

        assert (serial_number.empty() || serial_number == sn);
        serial_number = sn;
        // ---

        if (verbose > 0) 
            std::cout << "Processing (" << i + 1 << "/" << sequences.size() << "): " << it->first << std::endl;
        
        std::vector<cv::Mat> corners_aux;
        std::vector<int> fids;
        // if (modality_str == "Color")
        //     find_chessboard_corners<uls::ColorFrame>(it->second, pattern_size, corners_aux, fids, resize_dims, it->first, verbose > 1);
        // else if (modality_str == "Thermal")
        //     find_chessboard_corners<uls::ThermalFrame>(it->second, pattern_size, corners_aux, fids, resize_dims, it->first, verbose > 1);
        find_chessboard_corners(it->second, pattern_size, corners_aux, fids, resize_dims, it->first, verbose > 1);

        assert(corners_aux.size() == fids.size());

        cv::Mat corners (corners_aux.size(), pattern_size.height * pattern_size.width, CV_32FC2);
        std::vector<std::string> frames (corners_aux.size());
        for (int k = 0; k < corners_aux.size(); k++)
        {   
            corners_aux[k].reshape(2,1).copyTo(corners.row(k));
            frames[k] = it->second[fids[k]];
        }

        std::string seq_id = std::to_string(i);
        fstorage << ("sequence_dir-" + seq_id) << it->first;
        fstorage << ("corners-" + seq_id) << corners;
        fstorage << ("frames-" + seq_id) << frames;

        if (verbose > 0) 
            std::cout << "Corners found in: " << frames.size() << "/" << it->second.size() << " frames\n";
    }

    fstorage << "serial_number" << serial_number;

    fstorage.release();

    // // cv::Mat corners (sequence_corners.size(), pattern_size.height * pattern_size.width * 2, CV_32F);


    // // fstorage << "corners" << corners;
    // // fstorage << "frames" << corner_frames;

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

