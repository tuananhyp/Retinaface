#ifndef __ML_MODEL_MTCNN_FACE_H__
#define __ML_MODEL_MTCNN_FACE_H__

#include "ml_model_plugin.h"
#include "parameter_def.h"
#include <memory>

#include "ml_common.pb.h"
#include "mtcnn_faces.pb.h"
#include "ml_types.h"

namespace oclea
{

#define MAX_PNET_RESIZE_COUNT 16

typedef struct 
{
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
    uint32_t phy_addr;
    uint32_t full_width;
    uint32_t full_height;
    uint32_t pitch;
} roi_desc_t;

typedef struct {
    float x1, y1, x2, y2, score, left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y, left_mouth_x, left_mouth_y, right_mouth_x, right_mouth_y;
} BBox;

typedef struct
{
    float sc, ss, tx, ty;
} img_tool_similarity_transform_t;

typedef struct 
{
    char name[32];
    float v[128];
} face_feature_t;

// Model class implements IModelPlugin
class RetinaFaceModelClass : public IModelPlugin
{
public:
    RetinaFaceModelClass();
    virtual ~RetinaFaceModelClass();

    bool probe(const MODEL_INPUT_CONFIG &input_config,
               MODEL_OUTPUT_CONFIG &output_config);
    void set_image_tool(ImageToolPtr image_tool);
    void preprocess(const IMAGE_MEM_BLOCK &rgb_frame,
                    int               rgb_width,
                    int               rgb_height,
                    MlMetadata        metadata,
                    bool             &skip_frame);
    void postprocess(const vector<IMAGE_MEM_BLOCK> &outdata,
                     MlMetadata              metadata,
                     MODEL_TLV              &result);
    void pre_run_net(int               id,
                     MlMetadata        metadata,
                     vector<IMAGE_MEM_BLOCK> &indata,
                     int              &loop);
    void post_run_net(int                     id,
                      MlMetadata              metadata,
                      const vector<IMAGE_MEM_BLOCK> &outdata);
    void post_init_net(const vector<vector<IMAGE_MEM_BLOCK>> &net_in_data,
                       const vector<vector<IMAGE_MEM_BLOCK>> &net_out_data);
    MODEL_TYPE get_model_type(void);

private:
    bool _load_model_files();
    void _load_config();

    void generate_prior_boxes(float* anchors_out, int& num_anchors);

    void _generate_bboxes();

    void decode_boxes(float* decoded_boxes, float* loc, float* priors, int num_priors, float var0 = 0.1f, float var1 = 0.2f);
    void decode_landmarks(float* landmark, float* anchors, float* output, int num_priors, float variance0 = 0.1f);

    float IoU(const BBox& a, const BBox& b);
    void _nms(const float* boxes, const float* scores, const float* landmarks, int num_boxes, float iou_threshold, std::vector<int>& keep);

    void _generate_faces();
    void _prepare_mfacenet_input(vector<IMAGE_MEM_BLOCK> &indata);

    void _img_tool_setup_similarity_transform_5(img_tool_similarity_transform_t *tranform, float *cur_dim, float *fixed_dim);    
    float _inner_product_10(float *p1, float *p2);
    void _solve_linear_equations_4x4(float *result, float *matrix, float *target);
    int _most_significant_4(float v1, float v2, float v3, float v4);
    int _most_significant_3(float v1, float v2, float v3);
    int _most_significant_2(float v1, float v2);
    void _eq_5_multiply_sub(float *eq, float *eq_src, float v);
    void _eq_5_swap(float *eq, float *eq_src);
    void _img_tool_similarity_transform_u8(img_tool_similarity_transform_t *tranform,
        uint8_t *input, uint8_t *output,
        int input_width, int input_height,
        int output_width, int output_height,
        int input_pitch, int output_pitch);
    void _save_face_features(const IMAGE_MEM_BLOCK *rawdata);
    void _cache_thumbnail(MlFace *face);

private:
    MODEL_TYPE                              _model_type;
    string                                  _model_config_folder;
    string                                  _model_retinafacenet_file;
    string                                  _model_mfacenet_file;
    string                                  _model_folder;
    retina_face_parameter_t                 _model_params;
    ImageToolPtr                            _image_tool {NULL};
    bool                                    _is_dproc_init {false};

    IMAGE_MEM_BLOCK                         _rgb_frame;

    vector<IMAGE_MEM_BLOCK>                 _retinaface_indata;
    vector<IMAGE_MEM_BLOCK>                 _retinaface_outdata;
    float                                   _retinaface_nms_threshold;
    float                                   _retinaface_score_threshold;
    int                                     _retinaface_id;

    vector<BBox>                            bboxes;
    float *                                 scores;
    float *                                 anchors;
    int                                     num_anchors; 
    float *                                 decoded_boxes;
    float *                                 decoded_landmark; 
    float *                                 face_bboxes;
    float *                                 landmarks;
    vector<int>                             keep;
    bool                                    _mfacenet_enabled {true};
    int                                     _mfacenet_id;
    int                                     _mfacenet_max_count;
    int                                     _mfacenet_outdata_pitch;

    MtcnnFaces                              _faces;
    float                                   _face_width_threshold;

    struct timeval                          _detect_start_time;
    struct timeval                          _detect_end_time;
};

// Must register the model class
MODEL_PLUGIN_REGISTER(RetinaFaceModelClass);

}   //namespace oclea

#endif
