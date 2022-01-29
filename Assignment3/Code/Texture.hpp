//
// Created by LEI XU on 4/27/19.
//

#ifndef RASTERIZER_TEXTURE_H
#define RASTERIZER_TEXTURE_H
#include "global.hpp"
#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>
class Texture{
private:
    cv::Mat image_data;

public:
    Texture(const std::string& name)
    {
        image_data = cv::imread(name);
        cv::cvtColor(image_data, image_data, cv::COLOR_RGB2BGR);
        width = image_data.cols;
        height = image_data.rows;
    }

    int width, height;

    Eigen::Vector3f getColor(float u, float v)
    {
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        auto color = image_data.at<cv::Vec3b>(v_img, u_img);
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

    Eigen::Vector3f getColorBilinear(float u, float v)
    {
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        int u_img_0 = floor(u_img);
        int u_img_1 = std::min((int)ceil(u_img), width - 1);
        int v_img_0 = floor(v_img);
        int v_img_1 = std::min((int)ceil(v_img), height - 1);
        float u_ratio = (u_img - u_img_0);
        float v_ratio = (v_img - v_img_0);
        auto color00 = image_data.at<cv::Vec3b>(v_img_0, u_img_0);
        auto color10 = image_data.at<cv::Vec3b>(v_img_1, u_img_0);
        auto color0 = color00 * (1 - v_ratio) + color10 * v_ratio;

        auto color01 = image_data.at<cv::Vec3b>(v_img_0, u_img_1);
        auto color11 = image_data.at<cv::Vec3b>(v_img_1, u_img_1);
        auto color1 = color01 * (1 - v_ratio) + color11 * v_ratio;

        auto color = color0 * (1 - u_ratio) + color1 * u_ratio;

        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

};
#endif //RASTERIZER_TEXTURE_H
