#include "detect_node/box_position_publisher.hpp"

int main(int argc,char **argv)
{
    ros::init(argc, argv, "detect_node");

    ros::NodeHandle nh;
    char* model_path = (char*) "/home/cat/cbbhuxx/yolox_rknn_v2/src/detect_node/workspace/elite_sim.rknn";
    float box_conf_threshold = 0.7;
    bool imshow_is_show = true;

    Yolox_rknn yolo_rknn(nh, model_path, box_conf_threshold, imshow_is_show);

    ros::spin();
}
