#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <vector>
#include <thread>
#include <cstdlib>
#include <ctime>
#include "facedb.h"
#include <chrono>

using namespace std::chrono;
using namespace face;
using namespace std;

#define DEFAULT_CONFIG_FOLDER "/usr/share/ml-models/retina_face/"
#define DEFAULT_PLUGIN_LIBRARY "/usr/lib/ml-models/libmlmtcnnface.so"


int main(int argc, char** argv)
{

    string                 model_lib = DEFAULT_PLUGIN_LIBRARY;
    string                 model_config = DEFAULT_CONFIG_FOLDER;
    FaceDB                 facedb;

    
    facedb.initdb(model_lib, model_config);
    // start 
    auto start = high_resolution_clock::now();
    
    facedb.static_image(argv[1]);
    
    // stop 
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time execution: " << duration.count() / 1e6<< " s" << endl; 
    return 0;
}
