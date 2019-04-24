//
//  utils.hpp
//  RealsenseExamplesGettingStarted
//
//  Created by Albert Clapés on 27/11/2018.
//

#ifndef utils_synchronization_h
#define utils_synchronization_h

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>

namespace uls
{
    struct Timestamp
    {
        std::string id;
        int64_t time;
    };

    std::vector<std::string> tokenize(std::string s, std::string delimiter)
    {
        std::vector<std::string> tokens;

        size_t pos = 0;
        std::string token;
        while ((pos = s.find(delimiter)) != std::string::npos) 
        {
            token = s.substr(0, pos);
            tokens.push_back(token);
            s.erase(0, pos + delimiter.length());
        }
        tokens.push_back(s); // last token

        return tokens;
    }

    uls::Timestamp process_log_line(std::string line)
    {
        std::vector<std::string> tokens = tokenize(line, ",");

        uls::Timestamp ts;
        ts.id = tokens.at(0);
        std::istringstream iss (tokens.at(1));
        iss >> ts.time;
        
        return ts;
    }

    std::vector<uls::Timestamp> read_log_file(fs::path log_path)
    {
        std::ifstream log (log_path.string());
        std::string line;
        std::vector<uls::Timestamp> tokenized_lines;
        if (log.is_open()) {
            while (std::getline(log, line)) {
                try {
                    uls::Timestamp ts = process_log_line(line);
                    tokenized_lines.push_back(ts);
                }
                catch (std::exception & e)
                {
                    break;
                }
            }
            log.close();
        }

        return tokenized_lines;
    }

    void time_sync(std::vector<Timestamp> log_a, std::vector<Timestamp> log_b, std::vector<std::pair<Timestamp,Timestamp> > & log_synced, int64_t eps = 50, bool verbose = true)
    {
        std::vector<Timestamp> master;
        std::vector<Timestamp> slave; 
        if (log_a.size() < log_b.size())
        {
            master = log_a;
            slave  = log_b;
            if (verbose) std::cout << "a is master, b is slave\n";
        } else {
            master = log_b;
            slave  = log_a;
            if (verbose) std::cout << "b is master, a is slave\n";
        }

        int j = 0;

        for (int i = 0; i < master.size(); i++) 
        {
            Timestamp ts_m = master[i];
            std::vector<std::pair<int, int64_t> > matches;
            while (j < slave.size() && slave[j].time < ts_m.time + eps)
            {
                int64_t dist = abs(ts_m.time - slave[j].time);
                if (dist < eps)
                {
                    matches.push_back( std::pair<int,int64_t>(j, dist) );
                }
                j++;
            }

            if (!matches.empty())
            {
                std::pair<int, int64_t> m_best = matches[0];
                for (int k = 1; k < matches.size(); k++)
                {
                    if (matches[k].second < m_best.second)
                        m_best = matches[k];
                }

                std::pair<Timestamp,Timestamp> synced_pair;
                synced_pair.first  = log_a.size() < log_b.size() ? master[i] : slave[m_best.first];
                synced_pair.second = log_a.size() < log_b.size() ? slave[m_best.first] : master[i];
                log_synced.push_back(synced_pair);

                j = m_best.first; //+ 1;
            }

            // elif fill_with_previous:
            //     all_matches.append( ((i, ts_m), all_matches[-1][1]) ), ts_m in enumerate(master)
        }

        // for (auto log : log_synced)
        // {
        //     std::cout << log.first.time << "," << log.second.time << '\n';
        // }

        float dif_total = 0;
        if (verbose)
        {
            for (int i = 0; i < log_synced.size(); i++)
            {
                int64_t dif = log_synced[i].first.time - log_synced[i].second.time;
                std::cout << log_synced[i].first.time << ',' << log_synced[i].second.time << ',' << dif << '\n';
                dif_total += std::abs(dif);
            }
            std::cout << dif_total / log_synced.size() << std::endl;
        }
    }

    void time_sync(std::vector<Timestamp> log_a, std::vector<Timestamp> log_b, std::vector<std::pair<int,int> > & log_pairs, int64_t eps = 50, bool verbose = true)
    {
        log_pairs.clear();

        // std::vector<std::pair<Timestamp,Timestamp> > log_synced;

        std::vector<Timestamp> master;
        std::vector<Timestamp> slave;
        if (log_a.size() > log_b.size())
        {
            master = log_a;
            slave  = log_b;
            if (verbose) std::cout << "a is master, b is slave\n";
        } else {
            master = log_b;
            slave  = log_a;
            if (verbose) std::cout << "b is master, a is slave\n";
        }

        int j = 0;

        for (int i = 0; i < master.size(); i++) 
        {
            Timestamp ts_m = master[i];
            std::vector<std::pair<int, int64_t> > matches;
            while (j < slave.size() && slave[j].time < ts_m.time + eps)
            {
                int64_t dist = abs(ts_m.time - slave[j].time);
                if (dist < eps)
                {
                    matches.push_back( std::pair<int, int64_t>(j, dist) );
                }
                j++;
            }

            if (!matches.empty())
            {
                std::pair<int,int64_t> m_best = matches[0];
                for (int k = 1; k < matches.size(); k++)
                {
                    if (matches[k].second < m_best.second)
                        m_best = matches[k];
                }
                
                std::pair<int,int> synced_pair;
                synced_pair.first  = log_a.size() > log_b.size() ? i : m_best.first;
                synced_pair.second = log_a.size() > log_b.size() ? m_best.first : i;
                log_pairs.push_back(synced_pair);
            }
        }

        // if (verbose)
        // {
        //     for (int i = 0; i < log_synced.size(); i++)
        //     {
        //         std::cout << log_synced[i].first.time << "," << log_synced[i].second.time << '\n';
        //     }
        // }
    }
}

#endif // utils_synchronization_h