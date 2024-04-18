#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <ros/ros.h>
#include <set>
#include <vector>

#include "detect_node/box_position_publisher.hpp"

#ifndef _RKNN_POSTPROCESS_H_
#define _RKNN_POSTPROCESS_H_

#define OBJ_NAME_MAX_SIZE 8
#define OBJ_NUMB_MAX_SIZE 15
#define OBJ_CLASS_NUM     8
#define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM)

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct {
    BOX_RECT box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;



int post_process(rknn_output *outputs,int width, int height, rknn_tensor_attr* output_attrs, float conf_threshold, float nms_threshold, object_detect_result_list *od_results);

#endif

