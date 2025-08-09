/* This is adapted from Ambarella’s test_nnctrl_fdet */

#include "mobilefacenet.h"
#include "base64.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <json-c/json.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>

using namespace std;
using namespace cv;

namespace oclea
{

#define LANDMARK_NUM 10

MobileFaceNetModelClass::MobileFaceNetModelClass()
{
    _model_type = HW_ACCELERATED_MODEL;
}

MobileFaceNetModelClass::~MobileFaceNetModelClass()
{
}

bool MobileFaceNetModelClass::probe(const MODEL_INPUT_CONFIG &input_config,
                                MODEL_OUTPUT_CONFIG      &output_config)
{
    _model_config_folder = input_config.config_folder;
    _model_folder = _model_config_folder;

    output_config.model_type = _model_type;

    _mfacenet_id = 0;
    _mfacenet_max_count = _model_params.mfacenet_max_count;
    _load_model_files();

    NET_CONFIG cfg_mfacenet;
    cfg_mfacenet.id = _mfacenet_id;
    cfg_mfacenet.net_file = _model_mfacenet_file;
    cfg_mfacenet.input.clear();
    cfg_mfacenet.output.clear();
    cfg_mfacenet.input.push_back("data");
    cfg_mfacenet.output.push_back("fc5");
    cfg_mfacenet.net_loop_cnt = _mfacenet_max_count;
    output_config.net_configs.push_back(cfg_mfacenet);
    _mfacenet_id = cfg_mfacenet.id;

    return true;
}

void MobileFaceNetModelClass::set_image_tool(ImageToolPtr image_tool)
{
    _image_tool = image_tool;
}

void MobileFaceNetModelClass::set_meta_access(const IMetaReadAccess *meta_access)
{
    _meta_access = meta_access;
}

void MobileFaceNetModelClass::preprocess(const IMAGE_MEM_BLOCK &rgb_frame,
                                     int               rgb_width,
                                     int               rgb_height,
                                     MlMetadata        metadata,
                                     bool             &skip_frame)
{
    // gettimeofday(&_detect_start_time, NULL);
    _rgb_frame = rgb_frame;
    _rgb_frame.width = rgb_width;
    _rgb_frame.height = rgb_height;

    bool        success = true;
    MODEL_TLV   mtcnn_tlv;

    success = _meta_access->get_type(metadata,
                                     ML_MSG_RESULT_MTCNN_FACES,
                                     mtcnn_tlv);
    if (success) {
        success = _faces.ParseFromArray(mtcnn_tlv.value, mtcnn_tlv.length);
        if (!success) {
            printf("Failed to parse MTCNN faces info %u\n",
                   mtcnn_tlv.length);
            skip_frame = true;
        }
        else if (_faces.faces_size() == 0)
        {
            skip_frame = true;
        }
    }
    else
    {
        skip_frame = true;
    }
    
}

void MobileFaceNetModelClass::postprocess(const vector<IMAGE_MEM_BLOCK> &outdata,
                                      MlMetadata              metadata,
                                      MODEL_TLV              &result)
{
    int size = _faces.ByteSize();
    uint8_t *result_array = gst_ml_alloc_memory<uint8_t>(size);
    _faces.SerializeToArray(result_array, size);
    result.type = ML_MSG_RESULT_MOBILE_FACE_NET;
    result.length = size;
    result.value = result_array;

    gettimeofday(&_detect_end_time, NULL);
    static int n = 0;
    printf("[%d]Recognise %d face(s), elapsed time: %.3f\n", n++, _faces.faces_size(), (_detect_end_time.tv_sec - _detect_start_time.tv_sec) * 1000.0 + (_detect_end_time.tv_usec - _detect_start_time.tv_usec) / 1000.0);
    gettimeofday(&_detect_start_time, NULL);
}

void MobileFaceNetModelClass::pre_run_net(int               id,
                                      MlMetadata        metadata,
                                      vector<IMAGE_MEM_BLOCK> &indata,
                                      int              &loop)
{
    _prepare_mfacenet_input(indata);
    loop = _faces.faces_size();
}

void MobileFaceNetModelClass::post_run_net(int                     id,
                                       MlMetadata              metadata,
                                       const vector<IMAGE_MEM_BLOCK> &outdata)
{
    _save_face_features(&outdata[0]);
}

void MobileFaceNetModelClass::post_init_net(const vector<vector<IMAGE_MEM_BLOCK>> &net_in_data,
                                        const vector<vector<IMAGE_MEM_BLOCK>> &net_out_data)
{
     _mfacenet_outdata_pitch = net_out_data[_mfacenet_id][0].pitch;
}

IModelPlugin::MODEL_TYPE MobileFaceNetModelClass::get_model_type(void)
{
    return _model_type;
}

bool MobileFaceNetModelClass::_load_model_files()
{
    bool status = false;

    _model_mfacenet_file = _model_folder + "mobilefacenets_cavalry.bin";

    status = true;
    return status;
}

void MobileFaceNetModelClass::_prepare_mfacenet_input(vector<IMAGE_MEM_BLOCK> &indata)
{
    img_tool_similarity_transform_t tranform;
    float fixed_dim[LANDMARK_NUM * 2] = {
        30.2946, 51.6963,
        65.5318, 51.6963,
        48.0252, 71.7366,
        33.5493, 92.3655,
        62.7299, 92.3655};

    for (int i = 0; i < _faces.faces_size(); i++)
    {
        const MlFace face = _faces.faces(i);
        float landmarks[LANDMARK_NUM * 2] = {0};

        landmarks[0] = _rgb_frame.width * face.left_eye().x();
        landmarks[1] = _rgb_frame.height * face.left_eye().y();
        landmarks[2] = _rgb_frame.width * face.right_eye().x();
        landmarks[3] = _rgb_frame.height * face.right_eye().y();
        landmarks[4] = _rgb_frame.width * face.nose().x();
        landmarks[5] = _rgb_frame.height * face.nose().y();
        landmarks[6] = _rgb_frame.width * face.left_mouth().x();
        landmarks[7] = _rgb_frame.height * face.left_mouth().y();
        landmarks[8] = _rgb_frame.width * face.right_mouth().x();
        landmarks[9] = _rgb_frame.height * face.right_mouth().y();

        _img_tool_setup_similarity_transform_5(&tranform, landmarks, fixed_dim);
        uint8_t * output_addr = indata[0].virt_addr + i * indata[0].pitch * indata[0].height * 3;

        _img_tool_similarity_transform_u8(&tranform,
            (uint8_t *)(unsigned long)(_rgb_frame.virt_addr), output_addr,
            _rgb_frame.width, _rgb_frame.height,
            indata[0].width, indata[0].height,
            _rgb_frame.width, indata[0].pitch);
    }
}

void MobileFaceNetModelClass::_img_tool_setup_similarity_transform_5(img_tool_similarity_transform_t *tranform, float *cur_dim, float *fixed_dim)
{
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

float MobileFaceNetModelClass::_inner_product_10(float *p1, float *p2)
{
    int i = 0;
    float v = p1[0] * p2[0];
    for (i = 1; i < 10; i ++) {
        v += p1[i] * p2[i];
    }
    return v;
}

void MobileFaceNetModelClass::_solve_linear_equations_4x4(float *result, float *matrix, float *target)
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

int MobileFaceNetModelClass::_most_significant_4(float v1, float v2, float v3, float v4)
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

int MobileFaceNetModelClass::_most_significant_3(float v1, float v2, float v3)
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

int MobileFaceNetModelClass::_most_significant_2(float v1, float v2)
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

void MobileFaceNetModelClass::_eq_5_multiply_sub(float *eq, float *eq_src, float v)
{
    eq[0] = eq[0] - (eq_src[0] * v);
    eq[1] = eq[1] - (eq_src[1] * v);
    eq[2] = eq[2] - (eq_src[2] * v);
    eq[3] = eq[3] - (eq_src[3] * v);
    eq[4] = eq[4] - (eq_src[4] * v);
}

void MobileFaceNetModelClass::_eq_5_swap(float *eq, float *eq_src)
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

void MobileFaceNetModelClass::_img_tool_similarity_transform_u8(img_tool_similarity_transform_t *tranform,
    uint8_t *input, uint8_t *output,
    int input_width, int input_height,
    int output_width, int output_height,
    int input_pitch, int output_pitch)
{
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

void MobileFaceNetModelClass::_save_face_features(const IMAGE_MEM_BLOCK *rawdata)
{
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

void MobileFaceNetModelClass::_load_config()
{
    json_object *config = NULL;
    json_object *face_width_threshold = NULL;
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
    }
    json_object_put(config);
    free(data);
    data = NULL;
}

}   //namespace oclea

