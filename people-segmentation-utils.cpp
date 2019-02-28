#include <opencv2/opencv.hpp>
#include <iostream>

/*
 * Computes a segementation metric defined by the following formula:
 *      metric = alpha * (TP/(TP+FP+FN)) + (1-alpha) * (1/|TP|) \sum_{i=1}^{|TP|} IOU_i
 * where alpha is an experimental weighting factor and beta is a threshold over IoU measure in order to account for TPs.
 * 
 * The function expects G and P to be 1-channel integer matrices. Where different blobs are numbered from [1,N_G] and [1,N_P] respectively.
 * Zero (0) values in G and P are ignored.
 */

// void segmentation_metric(cv::Mat G, cv::Mat P, int & tp, int & fp, int & fn, float & term1, float & term2, float & metric, float alpha = 0.5, float beta = 0.5)
// {
//     // Assert G and P types. They must be one-channel integer matrices
//     assert(G.type() == CV_32SC1 && P.type() == CV_32SC1);

//     double min_g, max_g, min_p, max_p;
//     int min_idx, max_idx; // not used
//     cv::minMaxIdx(G, &min_g, &max_g, &min_idx, &max_idx);
//     cv::minMaxIdx(P, &min_p, &max_p, &min_idx, &max_idx);

//     tp = fp = fn = 0;

//     std::vector<int> markdown_g (max_g+1, 0);
//     std::vector<int> markdown_p (max_p+1, 0);
//     std::vector<float> tps_iou;
//     for (int i = min_g; i <= max_g; i++)
//     {
//         cv::Mat mGi = (G == i);
//         for (int j = min_p; markdown_g[i] < 1 && j <= max_p; j++)
//         {
//             if (markdown_p[j] < 1)
//             {
//                 cv::Mat mPj = (P == j);

//                 cv::Mat I, U;
//                 cv::bitwise_and(mGi, mPj, I);
//                 cv::bitwise_or(mGi, mPj, U);

//                 double iou = ((float) cv::countNonZero(I)) / cv::countNonZero(U);
//                 if (iou > beta)
//                 {
//                     markdown_g[i] = j;
//                     markdown_p[j] = i;
//                     tps_iou.push_back(iou);
//                 }
//             }
//         }
//     }

//     for (int i = 1; i <= max_g; i++)
//         if (markdown_g[i] > 0) tp++;
//         else fn++;

//     for (int j = 1; j <= max_p; j++)
//         if (markdown_p[j] < 1) 
//             fp++;

//     term1 = ((float) tp) / (tp+fp+fn);
//     term2 = 0.f;
//     for (int i = 0; i < tps_iou.size(); i++)
//         term2 += tps_iou[i];
//     term2 /= tps_iou.size();

//     metric = alpha * term1 + (1.f - alpha) * term2;
// }

void get_indices_map(cv::Mat M, std::map<int,int> & map, int init_value=0)
{
    // std::map<int, int> map;
    map.clear();
    std::map<int, int>::iterator it;
    for (int i = 0; i < M.rows; i++)
    {
        for (int j = 0; j < M.cols; j++)
        {  
            int idx = M.at<int>(i,j);
            map[idx] = init_value;
        }
    }
}

/*
 * Computes a segementation metric defined by the following formula:
 *      metric = alpha * (TP/(TP+FP+FN)) + (1-alpha) * (1/|TP|) \sum_{i=1}^{|TP|} IOU_i
 * where alpha is an experimental weighting factor and beta is a threshold over IoU measure in order to account for TPs.
 * 
 * The function expects G and P to be 1-channel integer matrices. Where different blobs are numbered and zero (0) values in G and P are ignored.
 */

void segmentation_metric(cv::Mat G, cv::Mat P, int & tp, int & fp, int & fn, float & term1, float & term2, float & metric, float alpha = 0.5, float beta = 0.5)
{
    // Assert G and P types. They must be one-channel integer matrices
    assert(G.type() == CV_32SC1 && P.type() == CV_32SC1);

    std::map<int,int> map_g, map_p;
    get_indices_map(G, map_g, 0);
    get_indices_map(P, map_p, 0);

    std::map<int,int>::iterator it;
    it = map_g.find(0);
    if (it != map_g.end()) map_g.erase(it);
    it = map_p.find(0);
    if (it != map_p.end()) map_p.erase(it);

    // std::vector<int> markdown_g (indices_G.size(), 0);
    // std::vector<int> markdown_p (indices_P.size(), 0);
    std::map<int,int>::iterator it_g, it_p;
    std::vector<float> tps_iou;
    for (it_g = map_g.begin(); it_g != map_g.end(); it_g++)
    {
        cv::Mat mask_g_i = (G == it_g->first);
        for (it_p = map_p.begin(); it_g->second < 1 && it_p != map_p.end(); it_p++)
        {
            if (it_p->second < 1)
            {
                cv::Mat mask_p_j = (P == it_p->first);

                cv::Mat I, U;
                cv::bitwise_and(mask_g_i, mask_p_j, I);
                cv::bitwise_or(mask_g_i, mask_p_j, U);

                double iou = ((float) cv::countNonZero(I)) / cv::countNonZero(U);
                if (iou > beta)
                {
                    it_g->second = it_p->first;
                    it_p->second = it_g->first;
                    tps_iou.push_back(iou);
                }
            }
        }
    }

    tp = fp = fn = 0;

    for (it = map_g.begin(); it != map_g.end(); it++)
        if (it->second > 0) tp++;
        else fn++;

     for (it = map_p.begin(); it != map_p.end(); it++)
        if (it->second == 0) fp++;

    term1 = ((float) tp) / (tp+fp+fn);

    if (tp == 0 && fp == 0 && fn == 0)
    {
        metric = 1.0;
    }
    else if (tp == 0)
    {
        metric = term1;
    }
    else
    {
        term2 = 0.f;
        for (int i = 0; i < tps_iou.size(); i++)
            term2 += tps_iou[i];
        term2 /= tps_iou.size();

        metric = alpha * term1 + (1.f - alpha) * term2;
    }
}

int main(int argc, char * argv[])
{
    cv::Mat G, P;
    int tp, fp, fn;
    float term1, term2, metric_result;
    float alpha = 0.5;
    float beta = 0.5;

    G = (cv::Mat_<int>(5,5) <<  0, 0, 0, 0, 2,
                                0, 0, 1, 0, 0,
                                0, 1, 1, 3, 0,
                                0, 0, 1, 0, 0,
                                0, 0, 1, 0, 0);

    P = (cv::Mat_<int>(5,5) <<  0, 0, 0, 0, 0,
                                1, 0, 2, 0, 0,
                                0, 0, 2, 3, 0,
                                0, 0, 2, 3, 6,
                                0, 0, 2, 0, 6);

    segmentation_metric(G, P, tp, fp, fn, term1, term2, metric_result, alpha, beta);
    std::cout << tp << ',' << fp << ',' << fn << ',' << term1 << ',' << term2 << ',' << metric_result << '\n'; // 1,3,2,0.166667,0.8,0.483333

    G = (cv::Mat_<int>(1,1) << 0);
    P = (cv::Mat_<int>(1,1) << 0);

    segmentation_metric(G, P, tp, fp, fn, term1, term2, metric_result, alpha, beta);
    std::cout << tp << ',' << fp << ',' << fn << ',' << term1 << ',' << term2 << ',' << metric_result << '\n'; // 0,0,0,-nan,0.8,1

    G = (cv::Mat_<int>(3,3) << 8, 8, 0,
                               0, 0, 0,
                               0, 0, 0);

    P = (cv::Mat_<int>(3,3) << 0, 0, 0,
                               0, 0, 0,
                               0, 8, 8);

    segmentation_metric(G, P, tp, fp, fn, term1, term2, metric_result, alpha, beta);
    std::cout << tp << ',' << fp << ',' << fn << ',' << term1 << ',' << term2 << ',' << metric_result << '\n'; // 0,0,0,-nan,0.8,1

    return 0;
}