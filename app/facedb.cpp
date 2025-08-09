#include "facedb.h"
#include "mlengine.h"
#include "ml_model_plugin.h"
#include "mtcnn_faces.pb.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "json-c/json.h"
#include "ml_meta.h"

using namespace cv;
using namespace oclea;
using namespace std;

namespace face
{

#define DEFAULT_RECOGNITION_PLUGIN "/usr/lib/ml-models/libmlmobilefacenet.so"

FaceDB::FaceDB()
{
}

FaceDB::~FaceDB()
{
    _close_vout();
}

int FaceDB::initdb(string model_plugin, string model_config)
{
    int status = 0;

    _model_plugin = model_plugin;
    _model_config = model_config;
    _model_plugin_recog = DEFAULT_RECOGNITION_PLUGIN;

    _input_config = {};

    _input_config.width = _facedb_param.face_sample_width;
    _input_config.height = _facedb_param.face_sample_height;
    _input_config.config_folder = model_config;
    _facedb_folder = _input_config.config_folder + "facedb";
    std::cout << "_facedb_folder " << _facedb_folder << std::endl;

    _load_config();

    std::cout << "starting to load faces to db " << std::endl;

    _load_faces_to_db();

    std::cout << "end of load faces to db " << std::endl;

    return status;
}

bool FaceDB::check_face(face_feature_t *face)
{
    float max_similarity = 0;
    float distance = 0;
    string name = "Unknown";

    {
        unique_lock<mutex>   lk(_userdb_mutex);
        for (auto &user : _userdb)
        {
            for (auto &f : user.faces)
            {
                float similarity = _cosine_similarity_quadratic((float *)&f.feature.v, face->v, 128);
                if (similarity > max_similarity)
                {
                    max_similarity = similarity;
                    name = user.name;
                }
            }
        }
    }

    distance = sqrt(max_similarity);
    if (distance > _facedb_param.similarity_threshold)
    {
        strcpy(face->name, name.c_str());
        //printf("Detect: %s, %f\n", face->name, distance);
    }
    else
    {
        strcpy(face->name, "Unknown");
        //printf("Detect: %s, %f\n", face->name, distance);
        return false;
    }

    return true;
}

int FaceDB::get_next_face_id()
{
    return _next_face_id++;
}

string FaceDB::get_facedb_folder()
{
    return _facedb_folder;
}

void FaceDB::_load_config()
{
    json_object *config = NULL;
    json_object *face_similarity_threshold = NULL;
    json_object *use_split_facenet = NULL;
    json_object *recognition_plugin = NULL;
    uint8_t *data = NULL;
    FILE *fp;
    string config_file = _model_config + "/config.json";
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
        if (json_object_object_get_ex(config, "face_similarity_threshold", &face_similarity_threshold))
        {
            double value = json_object_get_double(face_similarity_threshold);
            if (value > 0 && value <= 1)
            {
                _facedb_param.similarity_threshold = value;
            }
        }
        // face_recognition_mode can be 0/1/2, 0: disable recognition, 1: enable integrated recognition model, 2: enable separated recogition model
        if (json_object_object_get_ex(config, "face_recognition_mode", &use_split_facenet))
        {
            int value = json_object_get_int(use_split_facenet);
            _use_split_facenet = (value == 2 ? true : false);
        }
        if (json_object_object_get_ex(config, "recognition_plugin", &recognition_plugin))
        {
            const char *value = json_object_get_string(recognition_plugin);
            _model_plugin_recog = value;
        }
    }
    json_object_put(config);
    free(data);
    data = NULL;
}

void FaceDB::_load_faces_to_db()
{
    DIR *dp;
    struct dirent *dirp;

    if ((dp = opendir(_facedb_folder.c_str())) == NULL)
    {
        mkdir(_facedb_folder.c_str(), 0755);
        if ((dp = opendir(_facedb_folder.c_str())) == NULL)
        {
            printf("ERROR: cannot create face database folder\n");
            return;
        }
    }

    char facefolder[300] = {0};

    while((dirp = readdir(dp)) != NULL)
    {
        //Ignore reserved name
        if (strcmp(dirp->d_name, ".") == 0 || strcmp(dirp->d_name, "..") == 0 || strcmp(dirp->d_name, CACHED_THUMBNAIL_PATH) == 0)
            continue;
        sprintf(facefolder, "%s/%s", _facedb_folder.c_str(), dirp->d_name);

        //Loading user
        DIR *dp_f;
        struct dirent *dirp_f;
        if ((dp_f = opendir(facefolder)) == NULL)
            return;

        printf("Loading: %s\n", facefolder);

        facedb_user_t user;
        user.id = static_cast<uint32_t>(_userdb.size());
        user.name = dirp->d_name;
        user.datetime = _get_file_create_time(facefolder);

        FILE *fd_db;
        char facedbfile[512] = {0};
        sprintf(facedbfile, "%s/feature.db", facefolder);
        fd_db = fopen(facedbfile, "r");
        if (fd_db != NULL)
        {
            while (!feof(fd_db))
            {
                facedb_face_t face;
                if (fread(&face.feature, sizeof(face.feature), 1, fd_db) <= 0)
                {
                    break;
                }
                face.id = get_next_face_id();
                string thumbnail_file = _facedb_folder;
                thumbnail_file +="/";
                thumbnail_file += face.feature.thumbnail_path;
                face.datetime = _get_file_create_time(thumbnail_file);
                user.faces.push_back(face);
            }
            fclose(fd_db);
        }
        else
        {
            _mlengine = ml_engine_create(_model_plugin);
            std::cout << "created mlengine" << endl;
            _mlengine->start_engine(_input_config, &gst_ml_meta_filter_read_meta_access);
            std::cout << "started mlengine" << endl;

            if (_use_split_facenet)
            {
                _mlengine_recog = ml_engine_create(_model_plugin_recog);
                _mlengine_recog->start_engine(_input_config, &gst_ml_meta_filter_read_meta_access);
            }

            while((dirp_f = readdir(dp_f)) != NULL)
            {
                if (strcmp(dirp_f->d_name, ".") == 0 || strcmp(dirp_f->d_name, "..") == 0)
                    continue;
                _add_face_from_image(dirp->d_name, dirp_f->d_name, &user);
            }

            if (_use_split_facenet)
            {        
                _mlengine_recog->stop_engine();
                _mlengine_recog = nullptr;
            }
            _mlengine->stop_engine();
            _mlengine = nullptr;
        }
        closedir(dp_f);
        _userdb.push_back(user);
    }
    closedir(dp);
}

facedb_error_code FaceDB::add_face(uint32_t userid, uint32_t faceid, face_feature_t feature)
{
    unique_lock<mutex>   lk(_userdb_mutex);
    facedb_error_code ret = FACEDB_USER_NOT_FOUND;
    for (auto &user : _userdb)
    {
        if (user.id == userid)
        {
            facedb_face_t face;
            face.id = faceid;
            memcpy(&face.feature, &feature, sizeof(feature));

            string src_path = _facedb_folder + "/";
            string dst_path = _facedb_folder + "/";
            src_path += feature.thumbnail_path;
            dst_path += user.name;
            dst_path += strstr(feature.thumbnail_path, "/");
            string mv_cmd = "mv \"";
            mv_cmd += src_path;
            mv_cmd += "\" \"";
            mv_cmd += dst_path;
            mv_cmd += "\"";
            system(mv_cmd.c_str());
            face.datetime = _get_file_create_time(dst_path);
            
            memset(face.feature.thumbnail_path, 0, sizeof(face.feature.thumbnail_path));
            std::size_t found = dst_path.find_last_of("/");
            sprintf(face.feature.thumbnail_path, "%s/%s", user.name.c_str(), dst_path.substr(found + 1).c_str());
            _save_feature_to_file(user.name, face.feature);

            user.faces.push_back(face);
            ret = FACEDB_SUCCESS;
        }
    }
    return ret;
}

vector<facedb_user_t> FaceDB::get_users()
{
    unique_lock<mutex>   lk(_userdb_mutex);
    return _userdb;
}

facedb_error_code FaceDB::add_user(string username)
{
    facedb_error_code ret = FACEDB_SUCCESS;
    string user_folder = _facedb_folder + "/";
    user_folder += username;

    DIR *dp;
    if ((dp = opendir(user_folder.c_str())) == NULL)
    {
        if (mkdir(user_folder.c_str(), 0755) != 0)
        {
            ret = FACEDB_USER_NAME_INVALID;
            return ret;
        }
    }
    else
    {
        closedir(dp);
        ret = FACEDB_USER_NAME_EXISTS;
        return ret;
    }
    {
        unique_lock<mutex>   lk(_userdb_mutex);
        facedb_user_t newuser;
        newuser.id = static_cast<uint32_t>(_userdb.size());
        newuser.name = username;
        newuser.datetime = _get_file_create_time(user_folder);
        _userdb.push_back(newuser);
    }
    return ret;
}

facedb_error_code FaceDB::get_user(string username, facedb_user_t &user)
{
    unique_lock<mutex>   lk(_userdb_mutex);
    for (auto u : _userdb)
    {
        if (u.name == username)
        {
            user.id = u.id;
            user.name = u.name;
            user.datetime = u.datetime;
            user.faces = u.faces;
            return FACEDB_SUCCESS;
        }
    }
    return FACEDB_USER_NOT_FOUND;
}

facedb_error_code FaceDB::get_user(uint32_t userid, facedb_user_t &user)
{
    unique_lock<mutex>   lk(_userdb_mutex);
    for (auto u : _userdb)
    {
        if (u.id == userid)
        {
            user.id = u.id;
            user.name = u.name;
            user.datetime = u.datetime;
            user.faces = u.faces;
            return FACEDB_SUCCESS;
        }
    }
    return FACEDB_USER_NOT_FOUND;
}

facedb_error_code FaceDB::delete_user(uint32_t userid)
{
    unique_lock<mutex>   lk(_userdb_mutex);
    auto iter = _userdb.begin();

    while(iter != _userdb.end())
    {
        if (iter->id == userid)
        {
            string rm_cmd = "rm -rf \"";
            rm_cmd += _facedb_folder;
            rm_cmd += "/";
            rm_cmd += iter->name;
            rm_cmd += "\"";
            system(rm_cmd.c_str());
            _userdb.erase(iter);
            return FACEDB_SUCCESS;
        }
        iter++;
    }
    return FACEDB_USER_NOT_FOUND;
}

facedb_error_code FaceDB::delete_user_face(uint32_t userid, uint32_t faceid)
{
    facedb_error_code ret = FACEDB_USER_NOT_FOUND;
    unique_lock<mutex>   lk(_userdb_mutex);
    for (auto &user : _userdb)
    {
        if (user.id == userid)
        {
            ret = FACEDB_FACE_NOT_FOUND;
            bool found = false;
            auto iter = user.faces.begin();

            while (iter != user.faces.end())
            {
                if (iter->id == faceid)
                {
                    string rm_cmd = "rm -f \"";
                    rm_cmd += _facedb_folder;
                    rm_cmd += "/";
                    rm_cmd += iter->feature.thumbnail_path;
                    rm_cmd += "\"";
                    system(rm_cmd.c_str());
                    user.faces.erase(iter);
                    found = true;
                    break;
                }
                iter++;
            }

            if (found)
            {
                string rm_cmd = "rm -f ";
                rm_cmd += _facedb_folder;
                rm_cmd += "/";
                rm_cmd += user.name;
                rm_cmd += "/";
                rm_cmd += "feature.db";
                system(rm_cmd.c_str());

                for (auto face : user.faces)
                {
                    _save_feature_to_file(user.name, face.feature);
                }
                ret = FACEDB_SUCCESS;
                return ret;
            }
        }
    }
    return ret;
}

void FaceDB::_add_face_from_image(string name, string imgfile, facedb_user_t *user)
{
    facedb_face_t face;

    strcpy(face.feature.name, name.c_str());

    Mat orgimg;
    Mat resized_img;
    Mat yuvimg;

    char faceimage[512] = {0}; 
    sprintf(faceimage, "%s/%s/%s", _facedb_folder.c_str(), name.c_str(), imgfile.c_str());
    orgimg = imread(faceimage, IMREAD_COLOR);
    resized_img = _resize_image(orgimg);
    cvtColor(resized_img, yuvimg, COLOR_BGR2YUV_I420);

    MODEL_TLV result;
    bool deferred = false;
    _mlengine->process_frame(yuvimg.data, 0, nullptr, deferred, result);

    
    MtcnnFaces faces;

    if (!faces.ParseFromArray(result.value, result.length)) {
        return;
    }

    if (faces.faces_size() != 1)
    {
        printf("Ignore: only support 1 face (%d face detected)\n", faces.faces_size());
        return;
    }

    MlFace f;
    if (_use_split_facenet)
    {    
        GstMlMetaData metadata;
        metadata = gst_ml_meta_data_new();
        gst_ml_meta_data_add_type(&metadata, result);

        MODEL_TLV result_recog;
        MtcnnFaces faces_recog;

        _mlengine_recog->process_frame(resized_img.data, 0, &metadata, deferred, result_recog);
        gst_ml_meta_data_delete(&metadata);

        if (!faces_recog.ParseFromArray(result_recog.value, result_recog.length)) {
            return;
        }
        f = faces_recog.faces(0);
    }
    else
    {
        f = faces.faces(0);
    }

    if (f.feature_size() != 128)
    {
        printf("Ignore: Invalid feature size!\n");
        return;
    }
    for (int i = 0; i < f.feature_size(); i++)
    {
        face.feature.v[i] = f.feature(i);
    }
    face.id = get_next_face_id();
    sprintf(face.feature.thumbnail_path, "%s/%s", name.c_str(), imgfile.c_str());
    face.datetime = _get_file_create_time(faceimage);
    user->faces.push_back(face);

    _save_feature_to_file(name, face.feature);
    
    printf("Success.\n");
}

void FaceDB::_save_feature_to_file(string username, face_feature_t feature)
{
    FILE *fd_db;
    char facedbfile[512] = {0};
    sprintf(facedbfile, "%s/%s/feature.db", _facedb_folder.c_str(), username.c_str());
    fd_db = fopen(facedbfile, "a");
    if (fd_db != NULL)
    {
        fwrite(&feature, sizeof(feature), 1, fd_db);
        fclose(fd_db);
    }
}

float FaceDB::_cosine_similarity_quadratic(float *data_1, float *data_2, int n)
{
    int i = 0;
    float sum1 = 0;
    float sum2 = 0;
    float sum3 = 0;

    for (i = 0; i < n; i ++) {
        sum1 += data_1[i] * data_1[i];
        sum2 += data_2[i] * data_2[i];
        sum3 += data_1[i] * data_2[i];
    }

    if (0 > sum3) {
        return 0;
    }
    return ((sum3 * sum3) / (sum1 * sum2));
}

Mat FaceDB::_resize_image(const Mat& img)
{
    int width = img.cols;
    int height = img.rows;

    Mat target = Mat::zeros(_input_config.height, _input_config.width, img.type());
    float wh_ratio = (float)_input_config.width / (float)_input_config.height;

    float scale;
    cv::Rect roi;
    if (((float)width / (float)height) > wh_ratio)
    {
        scale = (float)_input_config.width / (float)width;
        roi.width = _input_config.width;
        roi.x = 0;
        roi.height = height * scale;
        roi.y = (_input_config.height - roi.height) / 2;
    }
    else
    {
        scale = (float)_input_config.height / (float)height;
        roi.y = 0;
        roi.height = _input_config.height;
        roi.width = width * scale;
        roi.x = (_input_config.width - roi.width) / 2;
    }

    resize(img, target(roi), roi.size());

    return target;
}



void FaceDB::_open_vout()
{
    if (!_is_vout_opened)
    {
        system("/bin/sh -c '/usr/bin/oclea_gstreamer_vout_example &'");
        _is_vout_opened = true;
    }
}

void FaceDB::_close_vout()
{
    if (_is_vout_opened)
    {
        system("killall -2 oclea_gstreamer_vout_example");
        _is_vout_opened = false;
    }
}

string FaceDB::_get_file_create_time(string path)
{
    string time_str;
    struct stat filestat;

    if (stat(path.c_str(), &filestat ) == 0)
    {
        time_str = ctime(&filestat.st_ctime);
    }
    return time_str;
}


void FaceDB::static_image(string imgfile)
{
    Mat yuvimg;

    Mat img_bgr = imread(imgfile, IMREAD_COLOR);

    Mat resized_img = _resize_image(img_bgr);
    cvtColor(resized_img, yuvimg, COLOR_BGR2YUV_I420);

    MODEL_TLV result;
    bool deferred = false;

    _mlengine = ml_engine_create(_model_plugin);
    std::cout << "create engine" << std::endl;

    _mlengine->start_engine(_input_config, &gst_ml_meta_filter_read_meta_access);
    _mlengine->process_frame(yuvimg.data, 0, nullptr, deferred, result);
    std::cout << "process_frame" << std::endl;

    _mlengine->stop_engine();
    _mlengine = nullptr;

    MtcnnFaces    faces;
    
    if (!faces.ParseFromArray(result.value, result.length)) {
        g_print("Failed to parse result!\n");
        return;
    }

    _faces.clear();

    for(int i = 0; i < faces.faces_size(); i++) {
        MlFace face = faces.faces(i);
        _faces.push_back(face);
    }
    _check_face();

    _draw_face(img_bgr);
}

void FaceDB::_check_face()
{
    for (auto & p : _faces)
    {
        float boxw = p.bbox().x_max() - p.bbox().x_min();
        
        if (boxw < _face_width_threshold)
        {
            p.set_score(0);
            continue;
        }

        
        face_feature_t ff;
        cout << "Value feature of p=" << p.feature_size() << endl;
        if (p.feature_size() != 128)
        {
            p.set_name("Unknown");
            continue;
        }
        for (int i = 0; i < p.feature_size(); i++)
        {
            ff.v[i] = p.feature(i);
        }
        // loop facebd to get name of face 
        check_face(&ff);
        p.set_name(ff.name);
    }
}

void FaceDB::_draw_face(Mat &image_bgr)
{
    unique_lock<mutex> lk(_faces_mutex);
    if (!_faces.empty())
    {
        for (auto & p : _faces)
        {
            Scalar bbox_color(0,255,0);

            if (p.score() == 0)
                continue;
	     
            int boxx = static_cast<int>(image_bgr.cols * p.bbox().x_min());
            int boxy = static_cast<int>(image_bgr.rows * p.bbox().y_min());
            int boxw = static_cast<int>(image_bgr.cols * (p.bbox().x_max() - p.bbox().x_min()));
            int boxh = static_cast<int>(image_bgr.rows * (p.bbox().y_max() - p.bbox().y_min()));
            rectangle(image_bgr, Rect(boxx, boxy, boxw, boxh), bbox_color, 2);
            
            Point origin(boxx, (boxy - 5));
            Scalar color(0, 0, 0);
            char text[256] = {0};
            sprintf(text, "%s[%f]", p.name().c_str(), p.score());
            // sprintf(text, "[%f]", p.score());

            putText(image_bgr, text, origin, FONT_HERSHEY_PLAIN, 1.0, color, 1, 8);
        }
        imwrite("image_out.png", image_bgr);
        std::cout << "saved Image" << endl;
    }
}

}  // namespace face



