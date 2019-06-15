//
//  src/utils/synchronization.hpp
//  Temporal synchronization of time series based on capturing time instants (expressed in milliseconds since Epoch)
//
//  Created by Albert Clapés on 27/11/2018.
//

#ifndef synchronization_hpp
#define synchronization_hpp

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>

namespace uls
{
    /* Struct definitions */;
    struct Timestamp
    {
        std::string id;
        int64_t time;
    };

    //
    // Reading timestamp log files
    //

    /*
     * Reads a log line.
     * @param s A log line
     * @param delimiter A delimiter
     */
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

    /*
     * Builds a timestamp struct from the log line
     * @param line A log line
     */
    uls::Timestamp line_to_timestamp(std::string line)
    {
        std::vector<std::string> tokens = tokenize(line, ",");

        uls::Timestamp ts;
        ts.id = tokens.at(0);
        std::istringstream iss (tokens.at(1));
        iss >> ts.time;
        
        return ts;
    }

    /*
     * Reads a log file containing frame ids and corresponding capturing times and generates a vector of timestamp structs.
     * @param log_path Path to log file
     */
    std::vector<uls::Timestamp> read_log(fs::path log_path)
    {
        std::ifstream log (log_path.string());
        std::string line;
        std::vector<uls::Timestamp> tokenized_lines;
        if (log.is_open()) {
            while (std::getline(log, line)) {
                try {
                    uls::Timestamp ts = line_to_timestamp(line);
                    tokenized_lines.push_back(ts);
                }
                catch (std::exception & e)
                {
                    return std::vector<uls::Timestamp>();
                }
            }
            log.close();
        }

        return tokenized_lines;
    }

    /*
     * Matches two series of timestamps.
     * 
     * Given two series of timestamps "a" and "b", it sets the shorter one as master and for each timestamp in the master looks for the
     * closest matching timestamp in the slave series (only considered if the difference between the two is smaller than a certain threshold 
     * value "eps".
     * @param a First log series
     * @param b Second log series
     * @param ab Output log series of matching timesamp (a_i,b_j) pairs
     * @param eps Threshold to consider a match (expressed in milliseconds)
     * @
     */
    void time_sync(std::vector<Timestamp> a, 
                   std::vector<Timestamp> b, 
                   std::vector<std::pair<Timestamp,Timestamp> > & ab, 
                   int64_t eps = 50,
                   bool verbose = true)
    {
        std::vector<Timestamp> master;
        std::vector<Timestamp> slave; 
        if (a.size() < b.size())
        {
            master = a;
            slave  = b;
        } else {
            master = b;
            slave  = a;
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
                synced_pair.first  = a.size() < b.size() ? master[i] : slave[m_best.first];
                synced_pair.second = a.size() < b.size() ? slave[m_best.first] : master[i];
                ab.push_back(synced_pair);

                j = m_best.first; //+ 1;
            }
        }

        if (verbose)
        {
            float dif_total = 0.f;
            for (int i = 0; i < ab.size(); i++)
            {
                int64_t dif = ab[i].first.time - ab[i].second.time;
                std::cout << ab[i].first.time << ',' << ab[i].second.time << ',' << dif << '\n';
                dif_total += std::abs(dif);
            }
            std::cout << "Synced sequences. Average frame difference: " << dif_total / ab.size() << std::endl;
        }
    }

    /*
     * Matches two series of timestamps.
     * 
     * Given two series of timestamps "a" and "b", it sets the shorter one as master and for each timestamp in the master looks for the
     * closest matching timestamp in the slave series (only considered if the difference between the two is smaller than a certain threshold 
     * value "eps".
     * @param a First log series
     * @param b Second log series
     * @param ab Output log series of matching timesamp (a_i,b_j) pairs
     * @param eps Threshold to consider a match (expressed in milliseconds)
     * @
     */
    void time_sync(std::vector<Timestamp> a, 
                   std::vector<Timestamp> b, 
                   std::vector<int> & indices_a, 
                   std::vector<int> & indices_b, 
                   int64_t eps = 50,
                   bool verbose = true)
    {
        indices_a.clear();
        indices_b.clear();

        std::vector<Timestamp> master;
        std::vector<Timestamp> slave; 
        if (a.size() < b.size())
        {
            master = a;
            slave  = b;
        } else {
            master = b;
            slave  = a;
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

                // std::pair<Timestamp,Timestamp> synced_pair;
                // synced_pair.first  = a.size() < b.size() ? master[i] : slave[m_best.first];
                // synced_pair.second = a.size() < b.size() ? slave[m_best.first] : master[i];
                // ab.push_back(synced_pair);
                a.size() < b.size() ? indices_a.push_back(i) : indices_b.push_back(m_best.first);
                a.size() < b.size() ? indices_b.push_back(m_best.first) : indices_a.push_back(i);

                j = m_best.first; //+ 1;
            }
        }

        if (verbose)
        {
            float diff_total = 0.f;
            for (int i = 0; i < indices_a.size(); i++)
            {
                int64_t a_time = a[indices_a[i]].time;
                int64_t b_time = b[indices_b[i]].time;
                int64_t diff = a_time - b_time;
                std::cout << a_time << ',' << b_time << ',' << diff << '\n';
                diff_total += std::abs(diff);
            }
            std::cout << "Synced sequences. Average frame difference: " << diff_total / indices_a.size() << std::endl;
        }
    }
}

#endif // synchronization_hpp