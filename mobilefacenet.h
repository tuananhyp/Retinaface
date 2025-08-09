#ifndef __ML_MODEL_MOBILE_FACE_NET_H__
#define __ML_MODEL_MOBILE_FACE_NET_H__

#include "ml_model_plugin.h"
#include "parameter_def.h"
#include <memory>

#include "ml_common.pb.h"
#include "mtcnn_faces.pb.h"
#include "ml_types.h"

namespace oclea
{

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
class MobileFaceNetModelClass : public IModelPlugin
{
public:
    MobileFaceNetModelClass();
    virtual ~MobileFaceNetModelClass();

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

    void set_meta_access(const IMetaReadAccess *meta_access);

private:
    bool _load_model_files();
    void _load_config();

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

private:
    MODEL_TYPE                              _model_type;
    string                                  _model_config_folder;
    vector<string>                          _model_pnet_files;
    string                                  _model_rnet_file;
    string                                  _model_onet_file;
    string                                  _model_mfacenet_file;
    string                                  _model_folder;
    retina_face_parameter_t                 _model_params;
    ImageToolPtr                            _image_tool {NULL};
    bool                                    _is_dproc_init {false};
    const IMetaReadAccess *                 _meta_access {NULL};

    IMAGE_MEM_BLOCK                         _rgb_frame;

    int                                     _mfacenet_id;
    int                                     _mfacenet_max_count;
    int                                     _mfacenet_outdata_pitch;

    MtcnnFaces                              _faces;
    float                                   _face_width_threshold;

    struct timeval                          _detect_start_time;
    struct timeval                          _detect_end_time;
};

// Must register the model class
MODEL_PLUGIN_REGISTER(MobileFaceNetModelClass);

}   //namespace oclea

#endif
