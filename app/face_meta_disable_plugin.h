#ifndef __MTCNN_META_DISABLE_PLUGIN_H__
#define __MTCNN_META_DISABLE_PLUGIN_H__

#include "ml_meta_plugin.h"
#include <memory>

#include "ml_common.pb.h"
#include "mtcnn_faces.pb.h"
#include "ml_types.h"

namespace oclea
{

// Meta plugin class implements IMetaPlugin
class MtcnnMetaDisableClass : public IMetaPlugin
{
public:
    MtcnnMetaDisableClass();
    virtual ~MtcnnMetaDisableClass();

    bool init(const META_INPUT_CONFIG &input_config,
              const IMetaWriteAccess  *meta_access,
              META_OUTPUT_CONFIG &output_config);
    void process_meta(const MODEL_FRAME_INFO &frame_info,
                      MlMetadata              metadata);

private:
    const IMetaWriteAccess *_meta_access;
};

// Must register the meta plugin class
META_PLUGIN_REGISTER(MtcnnMetaDisableClass);

}   //namespace oclea

#endif
