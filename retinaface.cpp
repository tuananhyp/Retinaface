#include "retinaface.h"
#include "base64.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <json-c/json.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <numeric>

using namespace std;
using namespace cv;
#define ROUND_UP_32(x) ((x)&0x1f ? (((x)&0xffffffe0) + 32) : (x))
namespace oclea
{

/**
 * @brief submodule to operate array data
 *
 */
#define FREE_MEM(m)     do { if(m != NULL) free(m); m = NULL;} while (0);

#define LANDMARK_NUM 10

#define MAX_ANCHORS 20000

const int IMAGE_SIZE = 640;
const int NUM_LAYERS = 3;
const int MIN_SIZES[NUM_LAYERS][2] = {{16, 32}, {64, 128}, {256, 512}};
const int NUM_MIN_SIZES[NUM_LAYERS] = {2, 2, 2};
const int STEPS[NUM_LAYERS] = {8, 16, 32};
const bool CLIP = false;

typedef enum 
{
    BBOX_STATUS_INIT,
    BBOX_STATUS_SELECTED,
    BBOX_STATUS_DISCARD,
} bbox_status_t;

RetinaFaceModelClass::RetinaFaceModelClass()
{
    _model_type = HW_ACCELERATED_MODEL;
}

RetinaFaceModelClass::~RetinaFaceModelClass()
{
    FREE_MEM(anchors);
    FREE_MEM(decoded_boxes);
    FREE_MEM(decoded_landmark);
    FREE_MEM(scores);
    FREE_MEM(face_bboxes);
    FREE_MEM(landmarks);
}

// done 
bool RetinaFaceModelClass::probe(const MODEL_INPUT_CONFIG &input_config,
                                MODEL_OUTPUT_CONFIG      &output_config)
{
    cout << "step 1" << endl; 
    size_t i;

    _model_config_folder = input_config.config_folder;
    _model_folder = _model_config_folder;

    output_config.model_type = _model_type;

    _retinaface_score_threshold = _model_params.retinaface_score_threshold;
    _retinaface_nms_threshold = _model_params.retinaface_nms_threshold;

    _mfacenet_max_count = _model_params.mfacenet_max_count;
    _face_width_threshold = _model_params.face_width_threshold;

    _load_model_files(); // 
    _load_config();

    NET_CONFIG cfg_retinaface;

    cfg_retinaface.id = i;
    cfg_retinaface.net_file = _model_retinafacenet_file;
    cfg_retinaface.input.clear();
    cfg_retinaface.output.clear();
    cfg_retinaface.input.push_back("input0");
    cfg_retinaface.output.push_back("output0");
    cfg_retinaface.output.push_back("593");
    cfg_retinaface.output.push_back("592");
    cfg_retinaface.net_loop_cnt = 0;
    output_config.net_configs.push_back(cfg_retinaface);
    _retinaface_id = cfg_retinaface.id;

    if (_mfacenet_enabled)
    {
        i++;

        NET_CONFIG cfg_mfacenet;

        cfg_mfacenet.id = i;
        cfg_mfacenet.net_file = _model_mfacenet_file;
        cfg_mfacenet.input.clear();
        cfg_mfacenet.output.clear();
        cfg_mfacenet.input.push_back("data");
        cfg_mfacenet.output.push_back("fc5");
        cfg_mfacenet.net_loop_cnt = _mfacenet_max_count;
        output_config.net_configs.push_back(cfg_mfacenet);
        _mfacenet_id = cfg_mfacenet.id;
    }
    return true;
}

// set_tool for process image 
void RetinaFaceModelClass::set_image_tool(ImageToolPtr image_tool)
{
    cout << "step 3" << endl;
    _image_tool = image_tool;
}

void RetinaFaceModelClass::preprocess(const IMAGE_MEM_BLOCK &rgb_frame,
                                     int               rgb_width,
                                     int               rgb_height,
                                     MlMetadata        metadata,
                                     bool             &skip_frame)
{
    cout << "step 2" << endl;
    // gettimeofday(&_detect_start_time, NULL);
    _rgb_frame = rgb_frame;
    _rgb_frame.width = rgb_width;
    _rgb_frame.height = rgb_height;
    _rgb_frame.pitch = ROUND_UP_32(rgb_width);
}

// done 
void RetinaFaceModelClass::postprocess(const vector<IMAGE_MEM_BLOCK> &outdata,
                                      MlMetadata              metadata,
                                      MODEL_TLV              &result)
{
    int size = _faces.ByteSize();
    uint8_t *result_array = gst_ml_alloc_memory<uint8_t>(size);
    _faces.SerializeToArray(result_array, size);
    result.type = ML_MSG_RESULT_MTCNN_FACES;
    result.length = size;
    result.value = result_array;

    // gettimeofday(&_detect_end_time, NULL);
    // static int n = 0;
    // printf("[%d]Detect %d face(s), elapsed time: %.3f\n", n++, _faces.faces_size(), (_detect_end_time.tv_sec - _detect_start_time.tv_sec) * 1000.0 + (_detect_end_time.tv_usec - _detect_start_time.tv_usec) / 1000.0);
}

// input for net
void RetinaFaceModelClass::pre_run_net(int               id,
                                      MlMetadata        metadata,
                                      vector<IMAGE_MEM_BLOCK> &indata,
                                      int              &loop)
{
    if (_mfacenet_enabled && id == _mfacenet_id)
    {
        cout << "step 5" << endl;
        _prepare_mfacenet_input(indata);
        cout << "step 19" << endl;

        loop = _faces.faces_size();
    }
    else if (id == _retinaface_id)
    {
        generate_prior_boxes(anchors, num_anchors);
        cout << "step 4" << endl;
        cout << "num_anchors=" << num_anchors << endl;
        
        _retinaface_indata[0].phy_addr = indata[0].phy_addr;
        _retinaface_indata[0].virt_addr = indata[0].virt_addr;
        _retinaface_indata[0].size = indata[0].size;

        _image_tool->resize_rgb(_rgb_frame, indata[0]);
    }
}

// post run net 
void RetinaFaceModelClass::post_run_net(int                     id,
                                       MlMetadata              metadata,
                                       const vector<IMAGE_MEM_BLOCK> &outdata)
{
    if (id == _retinaface_id)
    {
        cout << "step 6" << endl;
        cout << "size_outdata=" << outdata.size() << endl;
        for (size_t n = 0; n < outdata.size(); n++)
        {
            _retinaface_outdata[n].name = outdata[n].name;
            _retinaface_outdata[n].virt_addr = outdata[n].virt_addr;
            _retinaface_outdata[n].size = outdata[n].size;
            _retinaface_outdata[n].phy_addr = outdata[n].phy_addr;
        }
        
        _generate_bboxes();
        _nms(decoded_boxes, scores, decoded_landmark, num_anchors, _retinaface_nms_threshold, keep);
        _generate_faces();
    }
    else if (_mfacenet_enabled && id == _mfacenet_id)
    {
        cout << "step 7" << endl;
        _save_face_features(&outdata[0]);
    }
}

void RetinaFaceModelClass::post_init_net(const vector<vector<IMAGE_MEM_BLOCK>> &net_in_data,
                                        const vector<vector<IMAGE_MEM_BLOCK>> &net_out_data)
{
    cout << "step 8" << endl;
    IMAGE_MEM_BLOCK mdata;
   
    mdata.height = net_in_data[_retinaface_id][0].height;
    mdata.width = net_in_data[_retinaface_id][0].width;
    mdata.pitch = net_in_data[_retinaface_id][0].pitch;
    _retinaface_indata.push_back(mdata);
    

    cout << "debug" << net_out_data[_retinaface_id].size() << endl;
    for (size_t n = 0; n < net_out_data[_retinaface_id].size(); n++)
    {
        mdata.height = net_out_data[_retinaface_id][n].height;
        mdata.width = net_out_data[_retinaface_id][n].width;
        mdata.pitch = net_out_data[_retinaface_id][n].pitch;
        _retinaface_outdata.push_back(mdata);
    }

    anchors = (float *)malloc(MAX_ANCHORS * sizeof(float) * 4); // Anchors
    decoded_boxes = (float *)malloc(MAX_ANCHORS * sizeof(float) * 4); // Decoded boxes
    decoded_landmark = (float *)malloc(MAX_ANCHORS * sizeof(float) * 10); // Decoded landmarks
    scores = (float *)malloc(MAX_ANCHORS * sizeof(float)); // Scores
    face_bboxes = (float *)malloc(MAX_ANCHORS * sizeof(float) * 4); // Bboxes
    landmarks = (float *)malloc(MAX_ANCHORS * sizeof(float) * 10); // Landmarks
    
    if (decoded_boxes == NULL || decoded_landmark == NULL || scores == NULL || anchors == NULL)  
    {
        printf("Memory allocation error for RetinaFace outputs\n");
        return;
    }
    if (_mfacenet_enabled)
    {
        _mfacenet_outdata_pitch = net_out_data[_mfacenet_id][0].pitch;
    }
}

IModelPlugin::MODEL_TYPE RetinaFaceModelClass::get_model_type(void)
{    
    cout << "step 9" << endl;
    return _model_type;
}

bool RetinaFaceModelClass::_load_model_files()
{
    cout << "step 10" << endl;
    bool status = false;

    // load file weights 
    _model_retinafacenet_file = _model_folder + "/retinaface_cavalry.bin";
    _model_mfacenet_file = _model_folder + "/mobilefacenets_cavalry.bin";

    status = true;
    return status;
}

// done 
void RetinaFaceModelClass::generate_prior_boxes(float* anchors_out, int& num_anchors) {
    cout << "step 11" << endl;
    num_anchors = 0;
    
    for (int k = 0; k < NUM_LAYERS; ++k) {
        int step = STEPS[k];
        int feat_h = ceil((float)IMAGE_SIZE / step);
        int feat_w = ceil((float)IMAGE_SIZE / step);

        for (int i = 0; i < feat_h; ++i) {
            for (int j = 0; j < feat_w; ++j) {
                for (int m = 0; m < NUM_MIN_SIZES[k]; ++m) {
                    int min_size = MIN_SIZES[k][m];
                    float s_kx = (float)min_size / IMAGE_SIZE;
                    float s_ky = (float)min_size / IMAGE_SIZE;

                    float cx = (j + 0.5f) * step / IMAGE_SIZE;
                    float cy = (i + 0.5f) * step / IMAGE_SIZE;

                    if (CLIP) {
                        cx = min(1.0f, max(0.0f, cx));
                        cy = min(1.0f, max(0.0f, cy));
                        s_kx = min(1.0f, max(0.0f, s_kx));
                        s_ky = min(1.0f, max(0.0f, s_ky));
                    }

                    int idx = num_anchors * 4;
                    anchors_out[idx]     = cx;
                    anchors_out[idx + 1] = cy;
                    anchors_out[idx + 2] = s_kx;
                    anchors_out[idx + 3] = s_ky;

                    ++num_anchors;
                }
            }
        }
    }
}

void RetinaFaceModelClass::_generate_bboxes()
{
    cout << "step 12" << endl;
    cout << _retinaface_outdata[0].width << endl;    // phải là 4
    cout << _retinaface_outdata[1].width << endl;   // phải là 2
    cout << _retinaface_outdata[2].width << endl;    // phải là 10 
    
    float *bboxes_addr = (float *)(_retinaface_outdata[0].virt_addr);  
    float *score_addr  = (float *)(_retinaface_outdata[1].virt_addr);
    float *landmarks_addr   = (float *)(_retinaface_outdata[2].virt_addr);
    
    cout << "100 value landmmarks_addr" << endl;
    for(int i =0; i <100; i++)
    {
    	cout << landmarks_addr[i] << " "; 
    }
    cout << endl;
    for (int i = 0; i < num_anchors; i++)
    {
        scores[i] = score_addr[8*i+1];
    }
    
    for (int i = 0; i < num_anchors; i++)
    {
        face_bboxes[i * 4 + 0] = bboxes_addr[8*i+0]; // x_min
        face_bboxes[i * 4 + 1] = bboxes_addr[8*i+1]; // y_min
        face_bboxes[i * 4 + 2] = bboxes_addr[8*i+2]; // x_max
        face_bboxes[i * 4 + 3] = bboxes_addr[8*i+3]; // y_max
    }

    for (int i = 0; i < num_anchors; i++)
    {
        landmarks[i * 10 + 0] = landmarks_addr[16*i+0]; // x1
        landmarks[i * 10 + 1] = landmarks_addr[16*i+1]; // y1
        landmarks[i * 10 + 2] = landmarks_addr[16*i+2]; // x2
        landmarks[i * 10 + 3] = landmarks_addr[16*i+3]; // y2
        landmarks[i * 10 + 4] = landmarks_addr[16*i+4]; // x3
        landmarks[i * 10 + 5] = landmarks_addr[16*i+5]; // y3
        landmarks[i * 10 + 6] = landmarks_addr[16*i+6]; // x4
        landmarks[i * 10 + 7] = landmarks_addr[16*i+7]; // y4
        landmarks[i * 10 + 8] = landmarks_addr[16*i+8]; // x5
        landmarks[i * 10 + 9] = landmarks_addr[16*i+9]; // y5
    }

    decode_boxes(decoded_boxes, face_bboxes, anchors, num_anchors);          // Nx4  [0, 1]
    decode_landmarks(landmarks, anchors, decoded_landmark, num_anchors);     // Nx10
}

void RetinaFaceModelClass::decode_boxes(float* decoded_boxes, float* loc, float* priors, int num_priors, float var0, float var1)
{
    cout << "step 13" << endl;
    for (int i = 0; i < num_priors; ++i) {
        int offset = i * 4;

        // prior center, width, height
        float prior_cx = priors[offset];
        float prior_cy = priors[offset + 1];
        float prior_w  = priors[offset + 2];
        float prior_h  = priors[offset + 3];

        // loc offsets
        float loc_dx = loc[offset];
        float loc_dy = loc[offset + 1];
        float loc_dw = loc[offset + 2];
        float loc_dh = loc[offset + 3];

        // decode center
        float decode_cx = prior_cx + loc_dx * var0 * prior_w;
        float decode_cy = prior_cy + loc_dy * var0 * prior_h;

        // decode width & height
        float decode_w = prior_w * std::exp(loc_dw * var1);
        float decode_h = prior_h * std::exp(loc_dh * var1);

        // convert to [x_min, y_min, x_max, y_max]
        decoded_boxes[offset]     = std::min(1.0f, std::max(0.0f, decode_cx - decode_w / 2.0f));  // x_min
        decoded_boxes[offset + 1] = std::min(1.0f, std::max(0.0f, decode_cy - decode_h / 2.0f));  // y_min
        decoded_boxes[offset + 2] = std::min(1.0f, std::max(0.0f, decode_cx + decode_w / 2.0f));  // x_max
        decoded_boxes[offset + 3] = std::min(1.0f, std::max(0.0f, decode_cy + decode_h / 2.0f));  // y_max
    }
}
void RetinaFaceModelClass::decode_landmarks(float* landmark, float* anchors, float* output, int num_priors, float variance0) 
{
    cout << "step 14" << endl;
    for (int i = 0; i < num_priors; ++i) {
        float cx = anchors[i * 4 + 0];  // center x
        float cy = anchors[i * 4 + 1];  // center y
        float w  = anchors[i * 4 + 2];  // width
        float h  = anchors[i * 4 + 3];  // height

        for (int j = 0; j < 5; ++j) {
            float dx = landmark[i * 10 + j * 2 + 0];  // offset x
            float dy = landmark[i * 10 + j * 2 + 1];  // offset y

            output[i * 10 + j * 2 + 0] = cx + dx * variance0 * w;  // decoded x
            output[i * 10 + j * 2 + 1] = cy + dy * variance0 * h;  // decoded y
        }
    }
}

// Caculate IOU
float RetinaFaceModelClass::IoU(const BBox& a, const BBox& b) {
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);

    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter);
}



// Non-Maximum Suppression
void RetinaFaceModelClass::_nms(const float* boxes, const float* scores, const float* landmarks, int num_boxes, float iou_threshold, std::vector<int>& keep) 
{
    if (boxes == nullptr || scores == nullptr || num_boxes <= 0) {
        cerr << "Error: Invalid input to _nms" << endl;
        return;
    }

    cout << "step 15" << endl;

    // filter bounding box theo score
    
    for (int i = 0; i < num_boxes; ++i) {
        if (scores[i] < _retinaface_score_threshold) {
            continue;
        }
        BBox box;
        box.x1 = boxes[i * 4 + 0]*640;
        box.y1 = boxes[i * 4 + 1]*640;
        box.x2 = boxes[i * 4 + 2]*640;
        box.y2 = boxes[i * 4 + 3]*640;
        box.score = scores[i];
        box.left_eye_x = landmarks[i * 10 + 0];
        box.left_eye_y = landmarks[i * 10 + 1];
        box.right_eye_x = landmarks[i * 10 + 2];
        box.right_eye_y = landmarks[i * 10 + 3];
        box.nose_x = landmarks[i * 10 + 4];
        box.nose_y = landmarks[i * 10 + 5];
        box.left_mouth_x = landmarks[i * 10 + 6];
        box.left_mouth_y = landmarks[i * 10 + 7];
        box.right_mouth_x = landmarks[i * 10 + 8];
        box.right_mouth_y = landmarks[i * 10 + 9];
        bboxes.push_back(box);
    }

    vector<int> indices(bboxes.size());
    std::iota(indices.begin(), indices.end(), 0); // Khởi tạo [0, 1, 2, ...]
    std::sort(indices.begin(), indices.end(), [&](int i1, int i2) {
        return bboxes[i1].score > bboxes[i2].score;
    });

    // Non-Maximum Suppression
    const int MAX_KEEP = 300; 
    vector<bool> suppressed(bboxes.size(), false);
    keep.clear();

    for (int i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (suppressed[idx]) continue;

        keep.push_back(idx);
        if (keep.size() >= MAX_KEEP) break; 

        for (int j = i + 1; j < indices.size(); ++j) {
            int idx_j = indices[j];
            if (suppressed[idx_j]) continue;

            if (IoU(bboxes[idx], bboxes[idx_j]) > iou_threshold) {
                suppressed[idx_j] = true;
            }
        }
    }
}

void RetinaFaceModelClass::_generate_faces()
{
    cout << "step 16" << endl;
    _faces.clear_faces();
    for(auto idx : keep)
    {
        MlFace *face = _faces.add_faces();
        MlBbox *face_bbox = face->mutable_bbox();

        face_bbox->set_x_min(bboxes[idx].x1/640);
        face_bbox->set_y_min(bboxes[idx].y1/640);
        face_bbox->set_x_max(bboxes[idx].x2/640);
        face_bbox->set_y_max(bboxes[idx].y2/640);

        MlPoint *left_eye = face->mutable_left_eye();
        left_eye->set_x(bboxes[idx].left_eye_x);
        left_eye->set_y(bboxes[idx].left_eye_y);

        MlPoint *right_eye = face->mutable_right_eye();
        right_eye->set_x(bboxes[idx].right_eye_x);
        right_eye->set_y(bboxes[idx].right_eye_y);

        MlPoint *nose = face->mutable_nose();
        nose->set_x(bboxes[idx].nose_x);
        nose->set_y(bboxes[idx].nose_y);

        MlPoint *left_mouth = face->mutable_left_mouth();
        left_mouth->set_x(bboxes[idx].left_mouth_x);
        left_mouth->set_y(bboxes[idx].left_mouth_y);

        MlPoint *right_mouth = face->mutable_right_mouth();
        right_mouth->set_x(bboxes[idx].right_mouth_x);
        right_mouth->set_y(bboxes[idx].right_mouth_y);

        face->set_score(bboxes[idx].score);
        _cache_thumbnail(face);
    }
}

//done 
void RetinaFaceModelClass::_prepare_mfacenet_input(vector<IMAGE_MEM_BLOCK> &indata)
{
    cout << "step 17" << endl;
    img_tool_similarity_transform_t tranform;
    float fixed_dim[LANDMARK_NUM * 2] = {
        30.2946, 51.6963,
        65.5318, 51.6963,
        48.0252, 71.7366,
        33.5493, 92.3655,
        62.7299, 92.3655
    };
    cout << "_faces.faces_size()=" << _faces.faces_size() << endl; 

    for (int i = 0; i < _faces.faces_size(); i++)
    {
        
        const MlFace face = _faces.faces(i);
        float landmarkss[LANDMARK_NUM * 2] = {0};

        landmarkss[0] = _rgb_frame.width * face.left_eye().x();
        landmarkss[1] = _rgb_frame.height * face.left_eye().y();
        landmarkss[2] = _rgb_frame.width * face.right_eye().x();
        landmarkss[3] = _rgb_frame.height * face.right_eye().y();
        landmarkss[4] = _rgb_frame.width * face.nose().x();
        landmarkss[5] = _rgb_frame.height * face.nose().y();
        landmarkss[6] = _rgb_frame.width * face.left_mouth().x();
        landmarkss[7] = _rgb_frame.height * face.left_mouth().y();
        landmarkss[8] = _rgb_frame.width * face.right_mouth().x();
        landmarkss[9] = _rgb_frame.height * face.right_mouth().y();

        _img_tool_setup_similarity_transform_5(&tranform, landmarkss, fixed_dim);
        
        uint8_t * output_addr = indata[0].virt_addr + i * indata[0].pitch * indata[0].height * 3;      
    
        _img_tool_similarity_transform_u8(&tranform,
            (uint8_t *)(unsigned long)(_rgb_frame.virt_addr), output_addr,
            _rgb_frame.width, _rgb_frame.height,
            indata[0].width, indata[0].height,
            _rgb_frame.width, indata[0].pitch);
    }
}

//done 
void RetinaFaceModelClass::_img_tool_setup_similarity_transform_5(img_tool_similarity_transform_t *tranform, float *cur_dim, float *fixed_dim)
{
    cout << "step 20" << endl;
    int i = 0, j = 0;
    float matrix[4][4];
    float target[4];
    float col[4][10] = {
        {
            fixed_dim[0], fixed_dim[1], fixed_dim[2], fixed_dim[3], fixed_dim[4],
            fixed_dim[5], fixed_dim[6], fixed_dim[7], fixed_dim[8], fixed_dim[9],
        },
        {
            fixed_dim[1], 0 - fixed_dim[0], fixed_dim[3], 0 - fixed_dim[2], fixed_dim[5],
            0 - fixed_dim[4], fixed_dim[7], 0 - fixed_dim[6], fixed_dim[9], 0 - fixed_dim[8],
        },
        {
            1, 0, 1, 0, 1,
            0, 1, 0, 1, 0,
        },
        {
            0, 1, 0, 1, 0,
            1, 0, 1, 0, 1,
        },
    };

    //least squares solution
    for (i = 0; i < 4; i ++) {
        matrix[i][i] = _inner_product_10(col[i], col[i]);
        for (j = i + 1; j < 4; j ++) {
            matrix[j][i] = matrix[i][j] = _inner_product_10(col[i], col[j]);
        }
        target[i] = _inner_product_10(col[i], cur_dim);
    }

    _solve_linear_equations_4x4((float *) tranform, (float *) matrix, target);
    return;
}

//done
float RetinaFaceModelClass::_inner_product_10(float *p1, float *p2)
{
    int i = 0;
    float v = p1[0] * p2[0];
    for (i = 1; i < 10; i ++) {
        v += p1[i] * p2[i];
    }
    return v;
}

//done
void RetinaFaceModelClass::_solve_linear_equations_4x4(float *result, float *matrix, float *target)
{
    int i = 0;
    float eq[4][5];

    for (i = 0; i < 4; i ++) {
        eq[i][0] = matrix[0];
        eq[i][1] = matrix[1];
        eq[i][2] = matrix[2];
        eq[i][3] = matrix[3];
        eq[i][4] = target[i];
        matrix += 4;
    }

    i = _most_significant_4(eq[0][0], eq[1][0], eq[2][0], eq[3][0]);
    if (0 != i) {
        _eq_5_swap(eq[i], eq[0]);
    }

    for (i = 1; i < 4; i ++) {
        _eq_5_multiply_sub(eq[i], eq[0], eq[i][0] / eq[0][0]);
    }

    i = _most_significant_3(eq[1][1], eq[2][1], eq[3][1]);
    if (0 != i) {
        _eq_5_swap(eq[i + 1], eq[1]);
    }

    for (i = 2; i < 4; i ++) {
        _eq_5_multiply_sub(eq[i], eq[1], eq[i][1] / eq[1][1]);
    }

    i = _most_significant_2(eq[2][2], eq[3][2]);
    if (0 != i) {
        _eq_5_swap(eq[3], eq[2]);
    }

    _eq_5_multiply_sub(eq[3], eq[2], eq[3][2] / eq[2][2]);

    for (i = 0; i < 3; i ++) {
        _eq_5_multiply_sub(eq[i], eq[3], eq[i][3] / eq[3][3]);
    }

    for (i = 0; i < 2; i ++) {
        _eq_5_multiply_sub(eq[i], eq[2], eq[i][2] / eq[2][2]);
    }

    _eq_5_multiply_sub(eq[0], eq[1], eq[0][1] / eq[1][1]);

    result[0] = eq[0][4] / eq[0][0];
    result[1] = eq[1][4] / eq[1][1];
    result[2] = eq[2][4] / eq[2][2];
    result[3] = eq[3][4] / eq[3][3];
}

//done
int RetinaFaceModelClass::_most_significant_4(float v1, float v2, float v3, float v4)
{
    int most = 0;
    float mv = 0, mv_tmp = 0;

    if (0 < v1) {
        mv = v1;
    } else {
        mv = 0 - v1;
    }

    if (0 < v2) {
        mv_tmp = v2;
    } else {
        mv_tmp = 0 - v2;
    }

    if (mv_tmp > mv) {
        most = 1;
        mv = mv_tmp;
    }

    if (0 < v3) {
        mv_tmp = v3;
    } else {
        mv_tmp = 0 - v3;
    }

    if (mv_tmp > mv) {
        most = 2;
        mv = mv_tmp;
    }

    if (0 < v4) {
        mv_tmp = v4;
    } else {
        mv_tmp = 0 - v4;
    }

    if (mv_tmp > mv) {
        return 3;
    }
    return most;
}

//done
int RetinaFaceModelClass::_most_significant_3(float v1, float v2, float v3)
{
    int most = 0;
    float mv = 0, mv_tmp = 0;

    if (0 < v1) {
        mv = v1;
    } else {
        mv = 0 - v1;
    }

    if (0 < v2) {
        mv_tmp = v2;
    } else {
        mv_tmp = 0 - v2;
    }

    if (mv_tmp > mv) {
        most = 1;
        mv = mv_tmp;
    }

    if (0 < v3) {
        mv_tmp = v3;
    } else {
        mv_tmp = 0 - v3;
    }

    if (mv_tmp > mv) {
        return 2;
    }

    return most;
}

//done
int RetinaFaceModelClass::_most_significant_2(float v1, float v2)
{
    float mv = 0, mv_tmp = 0;

    if (0 < v1) {
        mv = v1;
    } else {
        mv = 0 - v1;
    }

    if (0 < v2) {
        mv_tmp = v2;
    } else {
        mv_tmp = 0 - v2;
    }

    if (mv_tmp > mv) {
        return 1;
    }

    return 0;
}
//done
void RetinaFaceModelClass::_eq_5_multiply_sub(float *eq, float *eq_src, float v)
{
    eq[0] = eq[0] - (eq_src[0] * v);
    eq[1] = eq[1] - (eq_src[1] * v);
    eq[2] = eq[2] - (eq_src[2] * v);
    eq[3] = eq[3] - (eq_src[3] * v);
    eq[4] = eq[4] - (eq_src[4] * v);
}
//done
void RetinaFaceModelClass::_eq_5_swap(float *eq, float *eq_src)
{
    float v = 0;

    v = eq[0];
    eq[0] = eq_src[0];
    eq_src[0] = v;

    v = eq[1];
    eq[1] = eq_src[1];
    eq_src[1] = v;

    v = eq[2];
    eq[2] = eq_src[2];
    eq_src[2] = v;

    v = eq[3];
    eq[3] = eq_src[3];
    eq_src[3] = v;

    v = eq[4];
    eq[4] = eq_src[4];
    eq_src[4] = v;
}
//done
void RetinaFaceModelClass::_img_tool_similarity_transform_u8(img_tool_similarity_transform_t *tranform,
    uint8_t *input, uint8_t *output,
    int input_width, int input_height,
    int output_width, int output_height,
    int input_pitch, int output_pitch)
{
    cout << "step 21" << endl;
    int i = 0, j = 0;
    float u, v, u_base, v_base;
    int ui, vi;
    uint8_t *input_1 = input + (input_pitch * input_height);
    uint8_t *input_2 = input_1 + (input_pitch * input_height);
    uint8_t *output_1 = output + (output_pitch * output_height);
    uint8_t *output_2 = output_1 + (output_pitch * output_height);

    if ((!tranform) || (!input) || (!output) || (!input_width) || (!input_height)
        || (!output_width) || (!output_height)
        || (!input_pitch) || (!output_pitch)
        || (input_pitch < input_width) || (output_pitch < output_width)) {
        return;
    }


    for (j = 0; j < output_height; j ++) {
        u_base = tranform->ss * ((float) j) + tranform->tx;
        v_base = tranform->sc * ((float) j) + tranform->ty;
        for (i = 0; i < output_width; i ++) {
            u = u_base + tranform->sc * ((float) i);
            v = v_base - tranform->ss * ((float) i);
            ui = (int) (u + 0.5);
            vi = (int) (v + 0.5);
            if (ui < 0) {
                ui = 0;
            } else if (ui >= ((float) input_width)) {
                ui = input_width - 1;
            }
            if (vi < 0) {
                vi = 0;
            } else if (vi >= ((float) input_height)) {
                vi = input_height - 1;
            }
            output[i] = input[vi * input_pitch + ui];
            output_1[i] = input_1[vi * input_pitch + ui];
            output_2[i] = input_2[vi * input_pitch + ui];
        }
        output += output_pitch;
        output_1 += output_pitch;
        output_2 += output_pitch;
    }

    return;
}
//done
void RetinaFaceModelClass::_save_face_features(const IMAGE_MEM_BLOCK *rawdata)
{
    cout << "step 22" << endl;
    if (rawdata == NULL)
        return;

    for (int i = 0 ; i < _faces.faces_size(); i++)
    {
        float *data = (float *)(unsigned long)(rawdata->virt_addr + i * _mfacenet_outdata_pitch);

        MlFace *face = _faces.mutable_faces(i);
        for (unsigned long n = 0; n < _mfacenet_outdata_pitch/sizeof(float); n++)
        {
            face->add_feature(data[n]);
        }
    }
}
//done
void RetinaFaceModelClass::_cache_thumbnail(MlFace *face)
{
    cout << "step 18" << endl;

    if ((face->bbox().x_max() - face->bbox().x_min()) < _face_width_threshold)
    {
        return;
    }


    int min_width = static_cast<int>(_rgb_frame.width * _face_width_threshold);
    if (min_width < 100)
        min_width = 100;
    roi_desc_t roi_desc;
    float scale = 1.2;  //Must > 1

    roi_desc.x = static_cast<uint32_t>(_rgb_frame.width * face->bbox().x_min());
    roi_desc.y = static_cast<uint32_t>(_rgb_frame.height * face->bbox().y_min());
    roi_desc.width = static_cast<uint32_t>(_rgb_frame.width * (face->bbox().x_max() - face->bbox().x_min()) * scale);
    roi_desc.height = static_cast<uint32_t>(_rgb_frame.height * (face->bbox().y_max() - face->bbox().y_min()) * scale);
   

    if (roi_desc.x > static_cast<uint32_t>(roi_desc.width * (scale - 1) / 2))
    {
        roi_desc.x -= static_cast<uint32_t>(roi_desc.width * (scale - 1) / 2);
    }
    else
    {
        roi_desc.x = 0;
    }
    if (roi_desc.y > static_cast<uint32_t>(roi_desc.height * (scale - 1) / 2))
    {
        roi_desc.y -= static_cast<uint32_t>(roi_desc.height * (scale - 1) / 2);
    }
    else
    {
        roi_desc.y = 0;
    }
    if (roi_desc.x + roi_desc.width > (uint32_t)_rgb_frame.width)
    {
        roi_desc.width = _rgb_frame.width - roi_desc.x;
    }

    if (roi_desc.y + roi_desc.height > (uint32_t)_rgb_frame.height)
    {
        roi_desc.height = _rgb_frame.height - roi_desc.y;
    }

    uint8_t *blue_result = (uint8_t *)_rgb_frame.virt_addr;
    uint8_t *green_result = (uint8_t *)(blue_result + _rgb_frame.height * _rgb_frame.width);
    uint8_t *red_result = (uint8_t *)(green_result + _rgb_frame.height * _rgb_frame.width);
    Mat blue = Mat(Size(_rgb_frame.width,_rgb_frame.height), CV_8UC1, blue_result)(Rect(roi_desc.x, roi_desc.y, roi_desc.width, roi_desc.height));
    Mat red = Mat(Size(_rgb_frame.width,_rgb_frame.height), CV_8UC1, red_result)(Rect(roi_desc.x, roi_desc.y, roi_desc.width, roi_desc.height));
    Mat green = Mat(Size(_rgb_frame.width,_rgb_frame.height), CV_8UC1, green_result)(Rect(roi_desc.x, roi_desc.y, roi_desc.width, roi_desc.height));
    Mat planes[] = {red, green, blue};
    Mat img;
    merge(planes, 3, img);
    if(img.empty())
    {
        cout << "img is empty" << endl;
        return;
    }
    Mat resized_img;
    resize(img, resized_img, Size(min_width, roi_desc.height * min_width / roi_desc.width));
    string *thumbnail = face->mutable_thumbnail();
    face->set_thumbnail_width(resized_img.cols);
    face->set_thumbnail_height(resized_img.rows);
    thumbnail->assign(Base64::Encode(resized_img.data, face->thumbnail_width() * face->thumbnail_height() * 3));

    // printf("==================================================\n");
    // printf("size=%lu\n", face->thumbnail().size());
    //printf("%f\n", img_str.size()*1.0/(img.cols * img.rows * 3));

    // char name[20] = {0};
    // static int num = 1;
    // sprintf(name, "/tmp/%d.jpg", num++);
    // imwrite(name, resized_img);
}

//done
void RetinaFaceModelClass::_load_config()
{
    json_object *config = NULL;
    json_object *face_width_threshold = NULL;
    json_object *mfacenet_enabled = NULL;
    uint8_t *data = NULL;
    FILE *fp;
    string config_file = _model_config_folder + "/config.json";
    fp = fopen(config_file.c_str(), "r");
    if (fp != NULL)
    {
        fseek(fp, 0L, SEEK_END);
        int filesize = ftell(fp);
        fseek(fp, 0L, SEEK_SET);
        data = (uint8_t *)malloc(filesize + 1);
        if (data != NULL)
        {
            fread(data, 1, filesize, fp);
        }
        fclose(fp);
    }
    if (data == NULL)
    {
        return;
    }
    config = json_tokener_parse((const char *)data);
    if (config != NULL)
    {
        if (json_object_object_get_ex(config, "face_width_threshold", &face_width_threshold))
        {
            double value = json_object_get_double(face_width_threshold);
            if (value >= 0 && value <= 1)
            {
                _face_width_threshold = value;
            }
        }
        // face_recognition_mode can be 0/1/2, 0: disable recognition, 1: enable integrated recognition model, 2: enable separated recogition model
        if (json_object_object_get_ex(config, "face_recognition_mode", &mfacenet_enabled))
        {
            int value = json_object_get_int(mfacenet_enabled);
            _mfacenet_enabled = (value != 1 ? false : true);
        }
    }
    json_object_put(config);
    free(data);
    data = NULL;
}

}   //namespace oclea

