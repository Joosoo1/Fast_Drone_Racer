#include "detect_node/box_position_publisher.hpp"
// #include "detect_node/my_opencl.hpp"
#include "detect_node/postprocess.hpp"
#include <cmath>
#include <fstream>


rknn_input inputs[1];
rknn_output outputs[3];
rknn_tensor_attr output_attrs[3];
object_detect_result_list od_results;
int K=0;

static const std::vector<std::string> CLASSES = {"frame", "left_top", "right_top", "left_bottom", "right_bottom", "origin", "frame_A", "frame_B"};


Yolox_rknn::Yolox_rknn(const ros::NodeHandle& nh, char* model_path, const float box_conf_threshold, bool imshow_is_show)
: nh_(nh), model_path_(model_path), box_conf_threshold_(box_conf_threshold), nms_threshold(0.45), imshow_is_show(imshow_is_show), color_sub(nh_, "/camera/color/image_raw", 1), depth_sub(nh_, "/camera/depth/image_rect_raw", 1), sync(MySyncPolicy(5),color_sub,depth_sub)
{
    // ---------------------rknn------------------------------
    if (!RKNN_init()) 
    {
        ROS_DEBUG("rknn init failed");
        return;
    }

    // ---------------------ros------------------------------
    target_pub = nh_.advertise<bbox_ex_msgs::BoundingBoxes>("/realsence/detector/target", 1);

    sync.registerCallback(boost::bind(&Yolox_rknn::target_callback, this, _1, _2));

    // color_sub= nh_.subscribe("/camera/color/image_raw", 1, &Yolox_rknn::target_callback, this);
}
Yolox_rknn::~Yolox_rknn(){
    ret = rknn_destroy(ctx);
    if(model_path_) free(model_path_);

    ret = rknn_outputs_release(ctx, 1, outputs);
}

void Yolox_rknn::dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

unsigned char* Yolox_rknn::load_data(FILE* fp, size_t ofst, size_t sz)
{
  unsigned char* data;
  int            ret;

  data = NULL;

  if (NULL == fp) {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0) {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char*)malloc(sz);
  if (data == NULL) {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}
unsigned char* Yolox_rknn::load_model(const char* filename, int* model_size)
{
    FILE*          fp;
    unsigned char* data;
    fp = fopen(filename, "rb");
    if (NULL == fp) 
    {
    printf("Open file %s failed.\n", filename);
    return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    data = load_data(fp, 0, size);
    fclose(fp);
    *model_size = size;
    return data;
}
bool Yolox_rknn::RKNN_info()
{
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) 
    {
        ROS_DEBUG("rknn_init error ret=%d\n", ret);
        return -1;
    }
    ROS_INFO("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) 
    {
        ROS_DEBUG("rknn_init error ret=%d\n", ret);
        return -1;
    }
    ROS_INFO("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for(uint32_t i = 0; i < io_num.n_input; i++) 
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) 
        {
            ROS_DEBUG("rknn_init error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }
        
    memset(output_attrs, 0, sizeof(output_attrs));
    for (uint32_t i = 0; i < io_num.n_output; i++) 
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) 
    {
        ROS_INFO("model is NCHW input fmt\n");
        channel = input_attrs[0].dims[1];
        height  = input_attrs[0].dims[2];
        width   = input_attrs[0].dims[3];
    }
    else 
    {
        ROS_INFO("model is NHWC input fmt\n");
        height  = input_attrs[0].dims[1];
        width   = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }

    ROS_INFO("model input height=%d, width=%d, channel=%d\n", height, width, channel);
}
bool Yolox_rknn::RKNN_init()
{   
    ROS_INFO("Loading model...\n");
    int            model_data_size = 0;
    unsigned char* model_data      = load_model(model_path_, &model_data_size);
    ret                            = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0) {
        ROS_DEBUG("rknn_init error ret=%d\n", ret);
        return -1;
    }
    // 获取相关信息
    RKNN_info();

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index        = 0;
    inputs[0].type         = RKNN_TENSOR_UINT8;
    inputs[0].size         = width * height * channel;
    inputs[0].fmt          = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 0;

    memset(&od_results, 0, sizeof(object_detect_result_list));

    return true;
}

bool bool_box(std::vector<object_detect_result> tag_boxes, object_detect_result frame_box, object_detect_result& result, int mode){
    for (auto& tag_box : tag_boxes){
        if (float(tag_box.box.left+tag_box.box.right)/2 < frame_box.box.left) {continue;}
        if (float(tag_box.box.left+tag_box.box.right)/2 > frame_box.box.right) {continue;}
        if (float(tag_box.box.top+tag_box.box.bottom)/2 < frame_box.box.top) {continue;}
        if (float(tag_box.box.top+tag_box.box.bottom)/2 > frame_box.box.bottom) {continue;}
        
        if (mode == 1) {
            if ((float(tag_box.box.bottom-tag_box.box.top) / float(frame_box.box.bottom-frame_box.box.top)) < 0.20){continue;}
            if ((float(tag_box.box.bottom-tag_box.box.top) / float(frame_box.box.bottom-frame_box.box.top)) > 0.24) {continue;}
        }
        else if (mode == 2){
            if ((float(tag_box.box.bottom-tag_box.box.top) / float(frame_box.box.bottom-frame_box.box.top)) < 0.233) {continue;}
            if ((float(tag_box.box.bottom-tag_box.box.top) / float(frame_box.box.bottom-frame_box.box.top)) > 0.273) {continue;}
        }
        else if (mode == 3){
            if ((float(tag_box.box.bottom-tag_box.box.top) / float(frame_box.box.bottom-frame_box.box.top)) < 0.295) {continue;}
            if ((float(tag_box.box.bottom-tag_box.box.top) / float(frame_box.box.bottom-frame_box.box.top)) > 0.335) {continue;}
        }
        else {
            if ((float(tag_box.box.bottom-tag_box.box.top) / float(frame_box.box.bottom-frame_box.box.top)) < 0.264) {continue;}
            if ((float(tag_box.box.bottom-tag_box.box.top) / float(frame_box.box.bottom-frame_box.box.top)) > 0.304) {continue;}
        }
        result = tag_box;
        return true;
    }
    return false;
}

bool doesFileExist(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();

}


void Yolox_rknn::target_callback(const sensor_msgs::ImageConstPtr& color_msg, const sensor_msgs::ImageConstPtr& depth_msg)
// void Yolox_rknn::target_callback(const sensor_msgs::ImageConstPtr& color_msg)
{   
    //K++;
    //std::string path = "/home/cat/cbbhuxx/yolox_rknn_v2/src/detect_node/workspace/val/" + std::to_string(K-1) + ".jpg";
    //if(!doesFileExist(path)){
    //    return;}    
    std::chrono::steady_clock::time_point Tbegin = std::chrono::steady_clock::now();
    //color_image = cv::imread(path);
    cv::Mat depth_image;
    color_image = cv_bridge::toCvCopy(color_msg, sensor_msgs::image_encodings::BGR8)->image;
    cv::Mat depth_image_big = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
    cv::Rect depth_rect(64,78,409,311);
    cv::Mat depth_image_temp = depth_image_big(depth_rect);
    cv::Size dstSize(640, 480);
    cv::resize(depth_image_temp, depth_image, dstSize, 0, 0);
    // preprocess
    cv::cvtColor(color_image, img, cv::COLOR_BGR2RGB);

    // inference
    inputs[0].buf = (void*)img.data;
    rknn_inputs_set(ctx, io_num.n_input, inputs);
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    
    // postprocess
    post_process(outputs, width, height, output_attrs, box_conf_threshold_, nms_threshold, &od_results);
    std::chrono::steady_clock::time_point Tend = std::chrono::steady_clock::now();
    // publish msg
    bbox_ex_msgs::BoundingBoxes bboxes_msg;
    
    std::vector<object_detect_result> frame_boxes;
    std::vector<object_detect_result> tag1_boxes;
    std::vector<object_detect_result> tag2_boxes;
    std::vector<object_detect_result> tag3_boxes;
    std::vector<object_detect_result> tag4_boxes;

    //std::ofstream outputFile("/home/cat/cbbhuxx/yolox_rknn_v2/src/detect_node/workspace/rknn_result/" + std::to_string(K-1) + ".txt");
    for (int i = 0; i < od_results.count; i++) {
        object_detect_result* det_result = &(od_results.results[i]);
        std::string class_id = CLASSES[det_result->cls_id];

        //if (!outputFile.is_open()) {
        //    std::cerr << "Error opening file for output!" << std::endl;
        //    return;
        //}
        //outputFile << class_id <<  " " << std::to_string(det_result->prop) << " " << std::to_string(det_result->box.left) << " " << std::to_string(det_result->box.top) << " " << std::to_string(det_result->box.right) << " " << std::to_string(det_result->box.bottom) << std::endl;

        if (det_result->cls_id == 0 || det_result->cls_id == 6 || det_result->cls_id == 7 || det_result->cls_id == 5)       {frame_boxes.push_back(*det_result);}
        else if (det_result->cls_id == 1)  {tag1_boxes.push_back(*det_result);}
        else if (det_result->cls_id == 2)  {tag2_boxes.push_back(*det_result);}
        else if (det_result->cls_id == 3)  {tag3_boxes.push_back(*det_result);}
        else if (det_result->cls_id == 4)   {tag4_boxes.push_back(*det_result);}                         

        if (imshow_is_show){
            cv::rectangle(color_image, cv::Point(det_result->box.left, det_result->box.top), cv::Point(det_result->box.right, det_result->box.bottom), cv::Scalar(0, 255, 0), 1);
            cv::putText(color_image, class_id, cv::Point(det_result->box.left, det_result->box.top), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1);
        }
    }
    //outputFile.close();
    for(auto& frame_box : frame_boxes){
        float depth = 0.0;
        float pix_x = (frame_box.box.left + frame_box.box.right) / 2.0;
        float pix_y = (frame_box.box.top + frame_box.box.bottom) / 2.0;
        bbox_ex_msgs::BoundingBox one_box;
        
        if (frame_box.cls_id==5) {depth = depth_image.at<unsigned short>(int(pix_y), int(pix_x));}
        else {
            std::vector<object_detect_result> temp_boxes;
            int all_exist = 0;
            object_detect_result result;
            object_detect_result result0;
            object_detect_result result1;
            object_detect_result result2;
            object_detect_result result3;
            if (bool_box(tag1_boxes, frame_box, result, 1)) {
                all_exist += 1;
                result0 = result;
                temp_boxes.push_back(result);}
            if (bool_box(tag2_boxes, frame_box, result, 2)) {
                all_exist += 1;
                result1 = result;
                temp_boxes.push_back(result);}
            if (bool_box(tag3_boxes, frame_box, result, 3)) {
                all_exist += 1;
                result2 = result;
                temp_boxes.push_back(result);}
            if (bool_box(tag4_boxes, frame_box, result, 4)) {
                all_exist += 1;
                result3 = result;
                temp_boxes.push_back(result);}
            if (all_exist == 4){
                if ((frame_box.box.bottom - frame_box.box.top) >= (frame_box.box.right - frame_box.box.left)){
                    if ((result2.box.bottom - result0.box.top) <= (result3.box.bottom - result1.box.top)){
                        one_box.angle = asin((frame_box.box.right - frame_box.box.left) / (frame_box.box.bottom - frame_box.box.top));}
                    else{
                        one_box.angle = -asin((frame_box.box.right - frame_box.box.left) / (frame_box.box.bottom - frame_box.box.top));}
                }
            }
            
            if (temp_boxes.size() == 0) {continue;}
            
            int init_num = temp_boxes.size();
            for(auto& temp_box : temp_boxes){
                std::vector<cv::KeyPoint> keypoints;
                cv::Mat grayImage;
                int left = std::max(int(temp_box.box.left), 0);
                int top = std::max(int(temp_box.box.top), 0);
                int right = std::min(int(temp_box.box.right), color_image.cols);
                int bottom = std::min(int(temp_box.box.bottom), color_image.rows);
                if ((right-left) < 5 || (bottom-top) < 5) {continue;}
                cv::Rect rect(left, top, right-left, bottom-top);
                cv::cvtColor(color_image(rect), grayImage, cv::COLOR_BGR2GRAY);
                fastDetector->detect(grayImage, keypoints);
                if (keypoints.size() < 1) {
                    init_num--;
                    continue;}

                float median = 0.0;
                std::vector<float> depth_ketpoints;
                for (auto& keypoint : keypoints){
                    depth_ketpoints.push_back(depth_image(rect).at<unsigned short>(keypoint.pt.y, keypoint.pt.x));}
                std::sort(depth_ketpoints.begin(), depth_ketpoints.end()); 
                if (keypoints.size() % 2 == 0) {
                    median = (depth_ketpoints[keypoints.size() / 2 - 1] + depth_ketpoints[keypoints.size() / 2]) / 2;} 
                else {
                    median = depth_ketpoints[keypoints.size() / 2];}
                if (median<100) {
                    init_num--;
                    continue;}
                depth += median;
            }
            if (init_num == 0) continue;
            depth = depth / init_num;
            if (depth/1000 < 0.05) continue;
        }

        one_box.class_id = frame_box.cls_id;
        one_box.center_z = depth/1000;
        one_box.center_x = float((pix_x - ppx) / fx * one_box.center_z);
        one_box.center_y = float((pix_y - ppy) / fy * one_box.center_z);
        bboxes_msg.bounding_boxes.push_back(one_box);
    }

    target_pub.publish(bboxes_msg);
    
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(Tend - Tbegin);
    ROS_INFO("YOLOX inference time: %f ms (%f fps)", time_used.count() * 1000, 1.0 / time_used.count());
    
    if (imshow_is_show){
        cv::imshow("YOLOX_ROS", color_image);
        cv::waitKey(1);
    }
}
