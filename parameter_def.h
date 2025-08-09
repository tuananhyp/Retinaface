#ifndef __ML_MODEL_PARAMETER_DEF_H__
#define __ML_MODEL_PARAMETER_DEF_H__

namespace oclea
{

typedef struct
{
    float       retinaface_nms_threshold {0.4};
    float       retinaface_score_threshold {0.6};

    int         mfacenet_max_count{150};

    float       face_width_threshold {0.01};
} retina_face_parameter_t;

}   //namespace oclea

#endif
 

