cmake_minimum_required(VERSION 3.5)

###############################################################################
#### INCLUDES #################################################################
###############################################################################
include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${ML_COMMON_INCLUDES}
                    ${MODEL_INCLUDES}
                    ${MODEL_RESULT_INCLUDE}
                    ${OPENCV_INCLUDES}
                    ${GSTREAMER_INCLUDES}
                    ${GST_RTSP_SERVER_INCLUDES}
                    ${OCLEA_PIPELINE_UTIL_INCLUDES}
                    ${ML_ENGINE_INCLUDE}
                    ${JSON_C_INCLUDES})

###############################################################################
#### DEPENDENCIES #############################################################
###############################################################################

set(GST_DEPS
    gstreamer-1.0
    gstbase-1.0
    gstvideo-1.0)

set(OCLEA_PIPELINE_UTIL_DEPS
    oclea_pipeline_util)

###############################################################################
#### GENERATE OUTPUT ##########################################################
###############################################################################

set(RETINAFACE_APP_SRC
    main.cpp
    facedb.cpp
)

add_executable(retinaface_detection ${RETINAFACE_APP_SRC})
target_link_libraries(retinaface_detection
                      mlengine
                      mlmeta
                      mlmtcnnproto
                      mlposeproto
                      gstrtspserver-1.0
                      json-c
                      oclea_overlay_drawer
                      ${OPENCV_DEPS}
                      ${GST_DEPS}
                      ${GLIB_DEPS}
                      ${PROTOBUF_DEPS}
                      ${CVFLOW_DEPS}
                      ${OCLEA_PIPELINE_UTIL_DEPS})
target_compile_options(retinaface_detection PRIVATE -Wno-unused-result)

install(TARGETS
        retinaface_detection
        RUNTIME DESTINATION ${INSTALL_PREFIX}usr/bin)


set(RETINAFACE_META_PLUGIN_DISABLE_SRC
    face_meta_disable_plugin.cpp)

add_library(mlmtcnnmetadisable SHARED ${RETINAFACE_META_PLUGIN_DISABLE_SRC})
target_link_libraries(mlmtcnnmetadisable PUBLIC
                      mlmtcnnproto
                      ${GLIB_DEPS}
                      ${PROTOBUF_DEPS})
install(TARGETS
        mlmtcnnmetadisable
        LIBRARY DESTINATION ${INSTALL_PREFIX}usr/lib/meta-plugins)
