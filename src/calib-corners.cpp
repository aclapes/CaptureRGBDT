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
        ("prefix,f", po::value<std::string>()->default_value(""), "Prefix")
        ("log-file,l", po::value<std::string>()->default_value(""), "Log file (e.g. rs.log)")
        ("file-ext,x", po::value<std::string>()->default_value(".jpg"), "Image file extension")
        ("pattern,p", po::value<std::string>()->default_value("11,8"), "Pattern size \"x,y\" squares")
        ("resize-dims,r", po::value<std::string>()->default_value("960,720"), "Resize frame to (h,w)")
         ("y-shift,y", po::value<int>()->default_value(0), "Y-shift")
        // ("verbose,v", po::bool_switch(&verbose), "Verbosity")
        ("verbose,v", po::value<int>()->default_value(0), "")
        ("input-list-file", po::value<std::string>(&input_list_file_str)->required(), "File containing list of calibration sequence directories")
        ("modality", po::value<std::string>(&modality_str)->required(), "Modality (either Color or Thermal)")
        ("output-file", po::value<std::string>(&output_file_str)->required(), "Output file");

    
    po::positional_options_description positional_options; 
    positional_options.add("input-list-file", 1); 
    positional_options.add("modality", 1); 
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

    verbose = vm["verbose"].as<int>();

    std::vector<std::string> pattern_size_aux;
    boost::split(pattern_size_aux, vm["pattern"].as<std::string>(), boost::is_any_of(","));
    assert(pattern_size_aux.size() == 2);

    int x = std::stoi(pattern_size_aux[0]);
    int y = std::stoi(pattern_size_aux[1]);
    assert(x > 2 && y > 2);
    cv::Size pattern_size (x,y);

    cv::Size resize_dims;
    if (vm.find("resize-dims") != vm.end())
    {
        std::vector<std::string> resize_dims_aux;
        boost::split(resize_dims_aux, vm["resize-dims"].as<std::string>(), boost::is_any_of(","));
        assert(resize_dims_aux.size() == 2);

        int w = std::stoi(resize_dims_aux[0]);
        int h = std::stoi(resize_dims_aux[1]);
        assert(w > 160 && h > 120);
        resize_dims = cv::Size(w,h); // resize frames to (w,h) before finding chessboard corners
    }
    
    // ---
    
    std::vector<std::string> sequence_dirs;

    fs::path input_p (input_list_file_str);
    if (fs::is_directory(input_p))
    {
        for(auto& entry : boost::make_iterator_range(fs::directory_iterator(input_p), {}))
            sequence_dirs.push_back(entry.path().string());
    }
    else
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
        std::cerr << "Calibration file (input-list-file argument) not found." << std::endl;
        return EXIT_FAILURE;
    }
    
    // fs::path input_dir (vm["input-dir"].as<std::string>());
    std::map<std::string, std::vector<std::string> > sequences;
    for (std::string seq_dir : sequence_dirs)
    {
        // std::vector<std::string> s = uls::list_files_in_directory(input_dir, vm["prefix"].as<std::string>(), vm["file-ext"].as<std::string>());
        // std::sort(s.begin(), s.end());
        std::vector<uls::Timestamp> log = uls::read_log_file(seq_dir + "/" + vm["log-file"].as<std::string>());
        std::vector<std::string> frame_paths (log.size());
        for (int i = 0; i < log.size(); i++)
        {
            uls::Timestamp ts = log[i];
            std::string rel_path = vm["prefix"].as<std::string>() + ts.id + vm["file-ext"].as<std::string>();
            // frame_paths[i] = seq_dir + "/" + rel_path;
            frame_paths[i] = rel_path;
        }
        sequences[seq_dir] = frame_paths;
    }

    // std::vector<cv::Mat> sequence_corners;
    // std::vector<std::string> corner_frames;


    cv::FileStorage fstorage (vm["output-file"].as<std::string>(), cv::FileStorage::WRITE);
    fstorage << "pattern_size" << pattern_size;
    fstorage << "resize_dims" << resize_dims;
    fstorage << "modality" << modality_str;
    fstorage << "y-shift" << vm["y-shift"].as<int>();
    fstorage << "prefix" << vm["prefix"].as<std::string>();
    fstorage << "log-file" << vm["log-file"].as<std::string>();
    fstorage << "file-extension" << vm["file-ext"].as<std::string>();
    fstorage << "sequence_dirs" << ((int) sequence_dirs.size());

    // // std::vector<int> fids; // keep track
    // // boost::progress_display pd (sequences.size());
    std::map<std::string, std::vector<std::string> >::iterator it;
    int i;
    for (it = sequences.begin(), i = 0; it != sequences.end(); it++, i++)
    {
        if (verbose > 0) 
            std::cout << "Processing (" << i + 1 << "/" << sequences.size() << "): " << it->first << std::endl;
        
        std::vector<cv::Mat> corners_aux;
        std::vector<int> fids;
        if (modality_str == "Color")
            uls::find_chessboard_corners<uls::ColorFrame>(it->second, pattern_size, corners_aux, fids, resize_dims, it->first, vm["y-shift"].as<int>(), verbose > 1);
        else if (modality_str == "Thermal")
            uls::find_chessboard_corners<uls::ThermalFrame>(it->second, pattern_size, corners_aux, fids, resize_dims, it->first, vm["y-shift"].as<int>(), verbose > 1);

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

