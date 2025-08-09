cmake_minimum_required(VERSION 3.5)

set(SUPPORTED_PLATFORMS cv22 cv25 cv28 cv2)
if(NOT ${PLATFORM} IN_LIST SUPPORTED_PLATFORMS)
  message(FATAL_ERROR "Platform '${PLATFORM}' is not supported; supported platforms are ${SUPPORTED_PLATFORMS}")
  return()
endif()

find_package(PkgConfig REQUIRED)
find_package(Protobuf REQUIRED)

pkg_check_modules(GLIB REQUIRED glib-2.0)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(JSONC REQUIRED json-c)
pkg_check_modules(OPENCV REQUIRED opencv4)

set(GLIB_INCLUDES ${GLIB_INCLUDE_DIRS})
set(GSTREAMER_INCLUDES ${GSTREAMER_INCLUDE_DIRS} ${GSTREAMER_BASE_INCLUDE_DIRS})
set(JSON_C_INCLUDES ${JSONC_INCLUDE_DIRS})
set(OPENCV_INCLUDES ${OPENCV_INCLUDE_DIRS})

set(MODEL_INCLUDES ${OCLEA_ML_COMMON_INCLUDES} ${OCLEA_GSTREAMER_ML_INCLUDES})

if (DEFINED ML_PLUGINS_IN_SDK)
    # If ML_PLUGINS_IN_SDK is set, install directly into SDK.
    # This is *not* set by default.
    set(INSTALL_PREFIX "")
else()
    # Install to a directory alongside the main fakeroot.
    set(INSTALL_PREFIX "../tsdk-ml-teknique-face/fakeroot/")
endif()

###############################################################################
#### INCLUDES #################################################################
###############################################################################

set(GLIB_DEPS
    pcre
    gio-2.0
    glib-2.0
    gmodule-2.0
    gobject-2.0
    blkid
    ffi
    mount
    uuid
    z)

set(OPENCV_DEPS
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_imgcodecs
    opencv_video
    opencv_videoio
    jpeg
    png)

set(PROTOBUF_DEPS
    protobuf
    z)

set(MODEL_RESULT_INCLUDE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${OCLEA_ML_POSENET_INCLUDES}
                    ${CMAKE_CURRENT_BINARY_DIR}
                    "${CMAKE_CURRENT_BINARY_DIR}/../tsdk-ml-teknique-posenet"
                    ${OPENCV_INCLUDES}
                    ${ML_COMMON_INCLUDES}
                    ${MODEL_INCLUDES}
                    ${JSON_C_INCLUDES}
                    ${GLIB_INCLUDES}
                    ${MODEL_RESULT_INCLUDE})

###############################################################################
#### DEPENDENCIES #############################################################
###############################################################################

###############################################################################
#### SOURCES ##################################################################
###############################################################################

set(Protobuf_IMPORT_DIRS ${ML_COMMON_PROTO_DIR})
protobuf_generate_cpp(MTCNN_PROTO_SRCS MTCNN_PROTO_HDRS
                      ${CMAKE_CURRENT_SOURCE_DIR}/include/mtcnn_faces.proto)

set(GSTREAMER_ML_PLUGIN_MODEL_MTCNNFACE_SRC
    retinaface.cpp
)
list(TRANSFORM GSTREAMER_ML_PLUGIN_MODEL_MTCNNFACE_SRC PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

set(GSTREAMER_ML_PLUGIN_MODEL_MOBILE_FACE_NET_SRC
    mobilefacenet.cpp
)
list(TRANSFORM GSTREAMER_ML_PLUGIN_MODEL_MOBILE_FACE_NET_SRC PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

###############################################################################
#### GENERATE OUTPUT ##########################################################
###############################################################################

add_library(mlmtcnnproto SHARED
            ${MTCNN_PROTO_SRCS})
target_link_libraries(mlmtcnnproto PUBLIC
                      mlprotocommon
                      ${PROTOBUF_DEPS})
set_target_properties(mlmtcnnproto PROPERTIES VERSION 1.0 SOVERSION 1)
install(TARGETS
        mlmtcnnproto
        LIBRARY DESTINATION usr/lib)

add_library(mlmtcnnface SHARED ${GSTREAMER_ML_PLUGIN_MODEL_MTCNNFACE_SRC})
target_link_libraries(mlmtcnnface PUBLIC
                      mlmtcnnproto
                      json-c
                      ${OPENCV_DEPS}
                      ${GLIB_DEPS}
                      ${AMBA_SSD_DEPS}
                      ${PROTOBUF_DEPS})
install(TARGETS
        mlmtcnnface
        LIBRARY DESTINATION ${INSTALL_PREFIX}usr/lib/ml-models)

add_library(mlmobilefacenet SHARED ${GSTREAMER_ML_PLUGIN_MODEL_MOBILE_FACE_NET_SRC})
target_link_libraries(mlmobilefacenet PUBLIC
                      mlmtcnnproto
                      json-c
                      ${GLIB_DEPS}
                      ${PROTOBUF_DEPS})
install(TARGETS
        mlmobilefacenet
        LIBRARY DESTINATION ${INSTALL_PREFIX}usr/lib/ml-models)

if(${PLATFORM} STREQUAL "cv28")
    file(GLOB ML_PLUGIN_MODEL_MTCNNFACE_MODEL_BIN "${CMAKE_CURRENT_SOURCE_DIR}/model/cv28/*.bin")
else()
    file(GLOB ML_PLUGIN_MODEL_MTCNNFACE_MODEL_BIN "${CMAKE_CURRENT_SOURCE_DIR}/model/*.bin")
endif()
install(FILES
        ${ML_PLUGIN_MODEL_MTCNNFACE_MODEL_BIN}
        DESTINATION ${INSTALL_PREFIX}usr/share/ml-models/retina_face)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/app")
