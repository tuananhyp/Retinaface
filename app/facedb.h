#ifndef __FACEDB_H__
#define __FACEDB_H__

#include <string>
#include <vector>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include "mlengine.h"
#include "face_parameter_def.h"

#include "oclea-ml-common/fps_counter.h"

#include <oclea-ml-posenet/pose.pb.h>
#include <oclea_gstreamer_pipeline.h>
#include <oclea_overlay_drawer.h>
#include "mtcnn_faces.pb.h"

using namespace cv;
namespace face
{
#define CACHED_THUMBNAIL_PATH           "cachedthumb"

typedef enum
{
    FACEDB_SUCCESS = 0,
    FACEDB_USER_NAME_EXISTS,
    FACEDB_USER_NAME_INVALID,
    FACEDB_USER_NOT_FOUND,
    FACEDB_FACE_NOT_FOUND
} facedb_error_code;

typedef struct 
{
    char name[32];
    float v[128];
    char thumbnail_path[128];
} face_feature_t;

typedef struct
{
    uint32_t                id;
    string                  datetime;
    face_feature_t          feature;
} facedb_face_t;

typedef struct
{
    uint32_t                id;
    string                  name;
    string                  datetime;
    vector<facedb_face_t>   faces;
} facedb_user_t;

class FaceDB
{
public:
    FaceDB();
    ~FaceDB();

    int initdb(string model_plugin, string model_config);
    bool check_face(face_feature_t *face);
    int get_next_face_id();
    string get_facedb_folder();

    void static_image(string imgfile);
    void get_confidence();

    vector<facedb_user_t> get_users();
    facedb_error_code add_user(string username);
    facedb_error_code add_face(uint32_t userid, uint32_t faceid, face_feature_t feature);
    facedb_error_code get_user(string username, facedb_user_t &user);
    facedb_error_code get_user(uint32_t userid, facedb_user_t &user);
    facedb_error_code delete_user(uint32_t userid);
    facedb_error_code delete_user_face(uint32_t userid, uint32_t faceid);

    
private:
    void _load_config();
    void _load_faces_to_db();
    void _add_face_from_image(string name, string filepath, facedb_user_t *user);
    void _save_feature_to_file(string username, face_feature_t feature);
    float _cosine_similarity_quadratic(float *data_1, float *data_2, int n);
    cv::Mat _resize_image(const cv::Mat& img);
    void _open_vout();
    void _close_vout();
    string _get_file_create_time(string path);
    void _check_face();
    void _draw_face(Mat &image_bgr);
private:
    string                            _model_plugin;
    string                            _model_config;
    string                            _model_plugin_recog;
    bool                              _use_split_facenet {false};

    facedb_param_t                    _facedb_param;
    shared_ptr<oclea::CMLEngine>      _mlengine {nullptr};
    shared_ptr<oclea::CMLEngine>      _mlengine_recog {nullptr};
    oclea::IModelPlugin::MODEL_INPUT_CONFIG  _input_config {};
    vector<facedb_user_t>             _userdb;
    string                            _facedb_folder;
    bool                              _is_vout_opened {false};
    uint32_t                          _next_face_id {0};
    mutex                             _userdb_mutex;
    float                       _face_width_threshold {0.03};
    int                         _overlay_width {1920};
    int                         _overlay_height {1080};
    
    vector<oclea::MlFace>             _faces;  
    std::mutex                  _faces_mutex;  
};

}  // namespace face

#endif  // __FACEDB_H__
