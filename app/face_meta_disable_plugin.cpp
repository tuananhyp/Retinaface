#include "face_meta_disable_plugin.h"

using namespace std;

namespace oclea
{

MtcnnMetaDisableClass::MtcnnMetaDisableClass()
{
}

MtcnnMetaDisableClass::~MtcnnMetaDisableClass()
{
}

bool MtcnnMetaDisableClass::init(const META_INPUT_CONFIG &input_config,
                                 const IMetaWriteAccess  *meta_access,
                                 META_OUTPUT_CONFIG      &output_config)
{
    _meta_access = meta_access;

    return true;
}

void MtcnnMetaDisableClass::process_meta(const MODEL_FRAME_INFO &frame_info,
                                         MlMetadata              metadata)
{
    bool        success = true;
    MODEL_TLV   mtcnn_tlv;
    MtcnnFaces  faces;
    MODEL_TLV   disable_tlv;

    success = _meta_access->get_type(metadata,
                                     ML_MSG_RESULT_MTCNN_FACES,
                                     mtcnn_tlv);

    if (!success) {
        /*
         * No MTCNN faces info found. This indicates this plugin is likely
         * being used wrong.
         */
        printf("No MTCNN faces result info found\n");
    }

    if (success) {
        success = faces.ParseFromArray(mtcnn_tlv.value, mtcnn_tlv.length);
        if (!success) {
            printf("Failed to parse MTCNN faces info %u\n",
                   mtcnn_tlv.length);
        }
    }

    if (success) {
        if (faces.faces_size() == 0) {
            /*
             * No faces found, so we want to add the disable metadata.
             */
            disable_tlv.type = ML_MSG_META_DISABLE;
            disable_tlv.length = 0;
            disable_tlv.value = nullptr;
            _meta_access->add_type(metadata, disable_tlv);
        }
    }
}

}   //namespace oclea

