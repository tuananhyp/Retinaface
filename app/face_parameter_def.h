#ifndef __ML_MODEL_PARAMETER_DEF_H__
#define __ML_MODEL_PARAMETER_DEF_H__

namespace face
{

typedef struct
{
    int         face_sample_width {320};
    int         face_sample_height {240};
    float       similarity_threshold {0.6};
} facedb_param_t;

}   //namespace face

#endif
