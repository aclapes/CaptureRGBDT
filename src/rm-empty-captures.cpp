#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/foreach.hpp>
#include <algorithm>
#include <iterator>
#include <filesystem>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

namespace 
{ 
  const size_t ERROR_IN_COMMAND_LINE = 1; 
  const size_t SUCCESS = 0; 
  const size_t ERROR_UNHANDLED_EXCEPTION = 2; 
 
} // namespace

int main(int argc, char * argv[])
{
    std::string input_dir_str;

    po::options_description desc("Program options");
    desc.add_options()
        ("help,h", "Print help messages")
        ("min-num-frames,m", po::value<int>()->default_value(10), "Minimum number of (depth) frames to keep the captured seq")
        ("input-dir", po::value<std::string>(&input_dir_str)->required(), "Input directory containing rs/pt frames and timestamp files");
    
    po::positional_options_description positional_options; 
    positional_options.add("input-dir", 1); 

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
    
    /*
     * CODE
     */

    fs::path input_dir (input_dir_str);
    if ( fs::is_directory(input_dir) )
    {
        for(auto& ei : boost::make_iterator_range(fs::directory_iterator(input_dir), {}))
        {
            if (fs::is_directory(ei))
            {
                bool need_to_rm = true;
                std::cout << "Analizing: " << ei.path() << std::endl;
                if (fs::is_directory(ei / fs::path("rs/")))
                {
                    if (fs::is_empty(ei / fs::path("rs/")))
                    {
                        std::cout << "Existing 'rs' folder is empty" << std::endl;
                    }
                    else if (fs::is_directory(ei / fs::path("rs/depth/")))
                    {
                        if (fs::is_empty(ei / fs::path("rs/depth/")))
                        {
                            std::cout << "Existing 'rs/depth' is empty" << std::endl;
                        }
                        else
                        {
                            int cnt = std::count_if(
                                fs::directory_iterator(ei / fs::path("rs/depth/")),
                                fs::directory_iterator(),
                                static_cast<bool(*)(const fs::path&)>(fs::is_regular_file) );
                            std::cout << "Found " << cnt << " files." << std::endl;

                            if (cnt >= vm["min-num-frames"].as<int>())
                                need_to_rm = false;
                        }
                    }
                    else
                    {
                        std::cout << "Does not contain 'rs/depth' folder" << std::endl;
                    }
                }
                else
                {
                    std::cout << "Does not contain 'rs' folder" << std::endl;
                }

                if (need_to_rm)
                {
                    fs::remove_all(ei);
                }
            }
        }
    }

    return SUCCESS;
}
