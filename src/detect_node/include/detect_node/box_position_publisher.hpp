#include <ros/ros.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <bbox_ex_msgs/BoundingBox.h>
#include <bbox_ex_msgs/BoundingBoxes.h>

#include "rknn_api.h"
// #include "rga.h"
// #include "im2d.h"

#pragma once

class Yolox_rknn
{
public:
    Yolox_rknn(const ros::NodeHandle& nh, char* model_path, const float box_conf_threshold, bool imshow_is_show);
    ~Yolox_rknn();
private:
    bool imshow_is_show;

    // ros parm
    ros::NodeHandle nh_;
    ros::Publisher target_pub;
    // ros::Subscriber color_sub;

    // rknn parm
    rknn_context   ctx;
    rknn_sdk_version version;
    rknn_input_output_num io_num;
    
    char*          model_path_;
    const float    nms_threshold;
    const float    box_conf_threshold_;
    int            ret;

    cv::Mat color_image, img, depth_image;
    
    int channel = 3;
    int width   = 0;
    int height  = 0;

    float ppx = 319.8746032714844;
    float ppy = 253.3507080078125;
    float fx   = 606.56494140625;
    float fy   = 606.4459228515625;

    cv::Ptr<cv::FastFeatureDetector> fastDetector = cv::FastFeatureDetector::create(95, true);
    message_filters::Subscriber<sensor_msgs::Image> color_sub;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
            sensor_msgs::Image> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync;
public:
    bool RKNN_init();
    unsigned char* load_model(const char* filename, int* model_size);
    bool RKNN_info();
    void target_callback(const sensor_msgs::ImageConstPtr& color_msg, const sensor_msgs::ImageConstPtr& depth_msg);
    // void target_callback(const sensor_msgs::ImageConstPtr& color_msg);
    void dump_tensor_attr(rknn_tensor_attr* attr);
    unsigned char* load_data(FILE* fp, size_t ofst, size_t sz);

};
