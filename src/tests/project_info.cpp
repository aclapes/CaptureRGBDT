#include <opencv2/opencv.hpp>
#include <iostream>

void three_part_text(cv::Mat & img,
                     std::vector<std::string> title_content,
                     std::vector<std::string> content,
                     std::vector<std::string> foot_content,
                     cv::Size s = cv::Size(1280,720),
                     double title_font_scale = 3,
                     double font_scale = 1.5,
                     double foot_font_scale = 1.5,
                     int font_thickness = 2,
                     int font_type = cv::FONT_HERSHEY_PLAIN,
                     cv::Scalar text_color = cv::Scalar(255,255,255),
                     cv::Scalar bg_color = cv::Scalar(0,0,0),
                     double interlinear = 2.0, 
                     double pad_left = 0.05)
{    
    img.create(s, CV_8UC3);
    img.setTo(bg_color);

    int height = s.height;
    int width = s.width;
    int baseline;
    cv::Size title_text_size, text_size, foot_font_size;

    title_text_size = cv::getTextSize(title_content[0], cv::FONT_HERSHEY_COMPLEX_SMALL, title_font_scale, font_thickness, &baseline);
    int title_height = interlinear * title_text_size.height * title_content.size();

    for (int i = 0; i < title_content.size(); i++)
    {
        cv::putText(img, title_content[i], cv::Point(pad_left*width, (i+1)*(title_height/title_content.size())),
            cv::FONT_HERSHEY_COMPLEX_SMALL, title_font_scale, text_color, font_thickness);
    }

    text_size = cv::getTextSize(content[0], cv::FONT_HERSHEY_COMPLEX_SMALL, font_scale, font_thickness, &baseline);
    int content_height = interlinear * text_size.height * content.size();

    for (int i = 0; i < content.size(); i++)
    {
        cv::putText(img, content[i], cv::Point(pad_left*width, height/2 - content_height/2 + i*(content_height/content.size())),
            cv::FONT_HERSHEY_COMPLEX_SMALL, font_scale, text_color, font_thickness);
    }

    foot_font_size = cv::getTextSize(foot_content[0], cv::FONT_HERSHEY_COMPLEX_SMALL, foot_font_scale, font_thickness, &baseline);
    int foot_height = interlinear * foot_font_size.height * foot_content.size();

    for (int i = 0; i < foot_content.size(); i++)
    {
        cv::putText(img, foot_content[i], cv::Point(pad_left*width, height - foot_height + i*(foot_height/foot_content.size())),
            cv::FONT_HERSHEY_COMPLEX_SMALL, font_scale, text_color, font_thickness);
    }
}

int main(int argc, char * argv[])
{   
    // cv::Mat img = cv::Mat::zeros(720, 1280, CV_8UC3);
    
    cv::Mat img;
    std::vector<std::string> title_content = {
        "SENIOR project (S&C and CVC)"
    };

    std::vector<std::string> content = {
        "Capturing data of (anonymized) people passing by",
        "to then train a Depth+Thermal human detector."
    };
    std::vector<std::string> foot_content = {
        "[+] Contact:", 
        " |- aclapes@cvc.uab.es",
        " '- sescalera@cvc.uab.es"
    };

    three_part_text(img, title_content, content, foot_content, cv::Size(640, 320), 1.5, 0.75, 0.75, 1);
    cv::imshow("info", img);
    cv::waitKey();

    return 0;
}

