#include "detect_node/postprocess.hpp"


static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}
inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}
static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        if (order[i] == -1 || classIds[i] != filterId)
        {
            continue;
        }
        int n = order[i];
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[i] != filterId)
            {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}
static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static int process_i8(int8_t *input, int grid_h, int grid_w, int height, int width, int stride,
                      std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                      int32_t zp, float scale)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);
    
    for (int i = 0; i < grid_h; ++i) {
        for (int j = 0; j < grid_w; ++j) {
            int8_t box_confidence = input[4 * grid_len + i * grid_w + j];
            
            if (box_confidence >= thres_i8) {
                int offset = i * grid_w + j;
                int8_t *in_ptr = input + offset;
                
                int8_t maxClassProbs = in_ptr[5 * grid_len];
                int maxClassId = 0;
                for (int k = 1; k < OBJ_CLASS_NUM; ++k)
                {
                    int8_t prob = in_ptr[(5 + k) * grid_len];
                    if (prob > maxClassProbs)
                    {
                        maxClassId = k;
                        maxClassProbs = prob;
                    }
                }

                if (maxClassProbs > thres_i8)
                {
                    float box_x = (deqnt_affine_to_f32(*in_ptr, zp, scale));
                    float box_y = (deqnt_affine_to_f32(in_ptr[grid_len], zp, scale));
                    float box_w = (deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale));
                    float box_h = (deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale));
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = exp(box_w) * stride;
                    box_h = exp(box_h) * stride;
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    objProbs.push_back((deqnt_affine_to_f32(maxClassProbs, zp, scale)) * (deqnt_affine_to_f32(box_confidence, zp, scale)));
                    classId.push_back(maxClassId);
                    validCount++;
                    boxes.push_back(box_x);
                    boxes.push_back(box_y);
                    boxes.push_back(box_w);
                    boxes.push_back(box_h);
                }
            }
        }
    }
    return validCount;
}

int post_process(rknn_output *outputs, int width, int height, rknn_tensor_attr* output_attrs, float conf_threshold, float nms_threshold, object_detect_result_list *od_results)
{
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = width;
    int model_in_h = height;

    memset(od_results, 0, sizeof(object_detect_result_list));

    for (int i = 0; i < 3; i++)
    {
        grid_h = output_attrs[i].dims[2];
        grid_w = output_attrs[i].dims[3];
        stride = model_in_h / grid_h;
        // std::cout << "grid_h = " << grid_h << std::endl;

        validCount += process_i8((int8_t *)outputs[i].buf, grid_h, grid_w, model_in_h, model_in_w, stride, filterBoxes, objProbs,
                                     classId, conf_threshold, output_attrs[i].zp, output_attrs[i].scale);
    }

    // no object detect
    if (validCount <= 0)
    {
        printf("warn: no object detected!\n");
        return 0;
    }
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i)
    {
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));

    for (auto c : class_set)
    {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    int last_count = 0;
    od_results->count = 0;

    /* box valid detect target */
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0];
        float y1 = filterBoxes[n * 4 + 1];
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        od_results->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w));
        od_results->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h));
        od_results->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w));
        od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h));
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;
    return 0;
}


