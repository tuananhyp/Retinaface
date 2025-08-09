#include <iostream>
#include <vector>

#include <cmath>
#include <numeric>
using namespace std;
#define MAX_ANCHORS 17000

const int IMAGE_SIZE = 640;
const int NUM_LAYERS = 3;
const int MIN_SIZES[NUM_LAYERS][2] = {{16, 32}, {64, 128}, {256, 512}};
const int NUM_MIN_SIZES[NUM_LAYERS] = {2, 2, 2};
const int STEPS[NUM_LAYERS] = {8, 16, 32};
const bool CLIP = false;
void generate_prior_boxes(float* anchors_out, int& num_anchors) {
    cout << "step 11" << endl;
    num_anchors = 0;
    
    for (int k = 0; k < NUM_LAYERS; ++k) {
        int step = STEPS[k];
        int feat_h = ceil((float)IMAGE_SIZE / step);
        int feat_w = ceil((float)IMAGE_SIZE / step);

        for (int i = 0; i < feat_h; ++i) {
            for (int j = 0; j < feat_w; ++j) {
                for (int m = 0; m < NUM_MIN_SIZES[k]; ++m) {
                    int min_size = MIN_SIZES[k][m];
                    float s_kx = (float)min_size / IMAGE_SIZE;
                    float s_ky = (float)min_size / IMAGE_SIZE;

                    float cx = (j + 0.5f) * step / IMAGE_SIZE;
                    float cy = (i + 0.5f) * step / IMAGE_SIZE;

                    if (CLIP) {
                        cx = min(1.0f, max(0.0f, cx));
                        cy = min(1.0f, max(0.0f, cy));
                        s_kx = min(1.0f, max(0.0f, s_kx));
                        s_ky = min(1.0f, max(0.0f, s_ky));
                    }

                    int idx = num_anchors * 4;
                    anchors_out[idx]     = cx;
                    anchors_out[idx + 1] = cy;
                    anchors_out[idx + 2] = s_kx;
                    anchors_out[idx + 3] = s_ky;

                    ++num_anchors;
                }
            }
        }
    }
}
int main() {
    float anchors_out[MAX_ANCHORS * 4];
    int num_anchors = 0;
    generate_prior_boxes(anchors_out, num_anchors);
    cout << "Number of anchors generated: " << num_anchors << endl;
    cout << "First 10 anchors:" << endl;
    for (int i = 0; i < 10; ++i) {
        cout << "Anchor " << i << ": (" 
             << anchors_out[i * 4] << ", " 
             << anchors_out[i * 4 + 1] << ", " 
             << anchors_out[i * 4 + 2] << ", " 
             << anchors_out[i * 4 + 3] << ")" << endl;
    }
    return 0;
}
