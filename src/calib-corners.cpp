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
                            //  std::string prefix = "",
                             std::function<cv::Mat(cv::Mat)> preprocessing_func = {},
                             bool verbose = true) 
{
    frames_corners.clear();
    frames_inds.clear(); 

    cv::Mat img, img_prev;
    cv::Mat corners, corners_prev;
    float tracking_enabled = false;       

    for (int i = 0; i < frames.size(); i++) 
    {
        // read and preprocess frame
        cv::Mat img = cv::imread(frames[i], cv::IMREAD_UNCHANGED);

        // depending on the modality you might need to preprocess the images somehow,
        // e.g. normalization and conversion to 8 bits
        if (preprocessing_func)
            img = preprocessing_func(img);
            
        // chessboard detection performs bad in very small resolution images. if that's the case
        // resize now (always after preprocessing to avoid 0-paddings in the normalization within
        // the preprocessing function
        uls::resize(img, img, resize_dims);

        cv::Mat img_gray;
        if (img.channels() == 3)
            cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
        else
            img_gray = img.clone();

        // find the corners
        corners.release();
        bool chessboard_found = findChessboardCorners(img, pattern_size, corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
        
        // if corners find...
        if (chessboard_found) 
        {
            cornerSubPix(img_gray, corners, cv::Size(21, 21), cv::Size(7, 7), cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 1e-1));
            tracking_enabled = true;
        }
        else if (tracking_enabled) // .. if not, try to track any previously detected ones
        {
            cv::Mat status, err;
            cv::calcOpticalFlowPyrLK(img_prev, img, corners_prev, corners, status, err, cv::Size(7,7));
            cornerSubPix(img_gray, corners, cv::Size(21, 21), cv::Size(7, 7), cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 1e-1));
            // error checking
            if ( ! uls::check_tracking_integrity(status, pattern_size) )  // none of the corners is lost
            {
                tracking_enabled = false;
                corners.release();
            }
        }

        if (tracking_enabled)
        {
            // the first boolean condition seems redundant to check_tracking_integrity(..) Leave it since it's not hurting
            // the second boolean condition transforms the tracked corners using a homography to a normalized 1x1 grid structure
            // and checks the transformed corners are coherent, e.g. (0,0), (0.1, 0), (0.2, 0) ... (1,0), (0, 0.1), (0.1, 0.1), ..., (1,1)
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
            if (img.channels() == 1)
                cv::cvtColor(img, cimg, cv::COLOR_GRAY2BGR);
            else
                cimg = img.clone();

            if (!corners.empty()) 
                cv::drawChessboardCorners(cimg, pattern_size, corners, chessboard_found);

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
        ("prefix,f", po::value<std::string>()->required(), "Prefix")
        ("output-file", po::value<std::string>(&output_file_str)->required(), "Output file")
        // ("log-file,l", po::value<std::string>()->default_value(""), "Log file (e.g. rs.log)")
        // ("file-ext,x", po::value<std::string>()->default_value(".jpg"), "Image file extension")
        ("pattern-size,p", po::value<std::string>()->default_value("11,8"), "Pattern size \"x,y\" squares")
        ("resize-dims,r", po::value<std::string>()->default_value("960,720"), "Resize frame to (h,w)")
        //  ("y-shift,y", po::value<int>()->default_value(0), "Y-shift")
        // ("verbose,v", po::bool_switch(&verbose), "Verbosity")
        ("verbose,v", po::value<int>()->default_value(0), "")
        // ("modality", po::value<std::string>(&modality_str)->required(), "Modality (either Color or Thermal)")
        ;

    
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
    for (std::string dir_path : sequence_dirs)
    {
        std::vector<std::string> s = uls::list_images_in_directory(dir_path, vm["prefix"].as<std::string>());  // lists jpg/png files
        std::sort(s.begin(), s.end());  // sort alphabetically
        sequences[dir_path] = s;
    }

    cv::FileStorage fstorage (vm["output-file"].as<std::string>(), cv::FileStorage::WRITE);
    fstorage << "nb_sequences" << ((int) sequence_dirs.size());
    fstorage << "prefix" << prefix;
    fstorage << "pattern_size" << pattern_size;
    fstorage << "resize_dims" << resize_dims;
    // fstorage << "modality" << vm["prefix"].as<std::string>(); //modality_str;
    // fstorage << "y-shift" << vm["y-shift"].as<int>();
    // fstorage << "log-file" << vm["log-file"].as<std::string>();
    // fstorage << "file-extension" << vm["file-ext"].as<std::string>();

    // // std::vector<int> frames_ids; // keep track
    // // boost::progress_display pd (sequences.size());
    std::map<std::string, std::vector<std::string> >::iterator it;
    int i;
    std::string serial_number;

    for (it = sequences.begin(), i = 0; it != sequences.end(); it++, i++)
    {
        // consistency check: same camera for all calibration sequences!
        std::string curr_serial_number;

        cv::FileStorage fs ( (fs::path(it->first) / fs::path("rs_info.yml")).string(), cv::FileStorage::READ );
        if (fs.isOpened()) 
        {
            fs["serial_number"] >> curr_serial_number;
            if (!serial_number.empty() && serial_number != curr_serial_number)
                return EXIT_FAILURE;

            serial_number = curr_serial_number;
        }

        if (verbose > 0) 
            std::cout << "Processing (" << i + 1 << "/" << sequences.size() << "): " << it->first << std::endl;
        
        // find corners
        std::vector<int> frames_ids;
        std::vector<cv::Mat> frames_corners;
        // find_chessboard_corners(it->second, pattern_size, frames_corners, frames_ids, resize_dims, verbose > 1);

        if (prefix == "rs/color")
            find_chessboard_corners(it->second, 
                                    pattern_size, 
                                    frames_corners, 
                                    frames_ids,
                                    resize_dims, 
                                    // static_cast<cv::Mat(*)(cv::Mat)>(&uls::color_to_8bit),
                                    std::function<cv::Mat(cv::Mat)>(), // preprocessing unneeded
                                    verbose > 1);
        else if (prefix == "pt/thermal")
            find_chessboard_corners(it->second, 
                                    pattern_size,
                                    frames_corners,
                                    frames_ids, 
                                    resize_dims, 
                                    static_cast<cv::Mat(*)(cv::Mat)>(&uls::thermal_to_8bit), // thermal normalization and conversion to 8-bit image
                                    verbose > 1);

        assert(frames_corners.size() == frames_ids.size());

        // data type transformation, more suitable to be stored in a YAML using a cv::FileStorage object
        cv::Mat corners (frames_corners.size(), pattern_size.height * pattern_size.width, CV_32FC2);
        std::vector<std::string> frames_paths (frames_corners.size());
        for (int k = 0; k < frames_corners.size(); k++)
        {   
            frames_corners[k].reshape(2,1).copyTo(corners.row(k)); // reshape to 2 channels, 1 column (the number of rows is inferred)
            
            if (fs::is_directory(input_p))
                frames_paths[k] = fs::relative(it->second[frames_ids[k]], input_p).string();
            else
                frames_paths[k] = it->second[frames_ids[k]];
        }

        std::string id = std::to_string(i);
        fstorage << ("sequence_dir-" + id) << (fs::is_directory(input_p) ? fs::relative(it->first, input_p).string() : it->first);
        fstorage << ("corners-" + id) << corners;
        fstorage << ("frames-" + id)  << frames_paths;

        if (verbose > 0) 
            std::cout << "Total frames containing corners: " << frames_paths.size() << "/" << it->second.size() << "\n";
    }

    fstorage << "serial_number" << serial_number;
    fstorage.release();

    return EXIT_SUCCESS;
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

