// Online C++ compiler to run C++ program online
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace std;

#define _retinaface_score_threshold  0.4
typedef struct {
    float x1, y1, x2, y2, score;
} BBox;

float IoU(const BBox& a, const BBox& b) {
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);

    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter);
}

void _nms(const float* boxes, const float* scores, int num_boxes, float iou_threshold, std::vector<int>& keep) 
{
    if (boxes == nullptr || scores == nullptr || num_boxes <= 0) {
        cerr << "Error: Invalid input to _nms" << endl;
        return;
    }

    cout << "step 15" << endl;

    // filter bounding box theo score
    std::vector<BBox> bboxes;
    for (int i = 0; i < num_boxes; ++i) {
        if (scores[i] < _retinaface_score_threshold) {
            continue;
        }
        BBox box;
        box.x1 = boxes[i * 4 + 0]*640;
        box.y1 = boxes[i * 4 + 1]*640;
        box.x2 = boxes[i * 4 + 2]*640;
        box.y2 = boxes[i * 4 + 3]*640;
        box.score = scores[i];
        bboxes.push_back(box);
    }

    cout << "Number of bounding boxes after score filtering: " << bboxes.size() << endl;

    vector<int> indices(bboxes.size());
    std::iota(indices.begin(), indices.end(), 0); // Khởi tạo [0, 1, 2, ...]
    std::sort(indices.begin(), indices.end(), [&](int i1, int i2) {
        return bboxes[i1].score > bboxes[i2].score;
    });

    // Non-Maximum Suppression
    const int MAX_KEEP = 300; 
    vector<bool> suppressed(bboxes.size(), false);
    keep.clear();

    for (int i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (suppressed[idx]) continue;

        keep.push_back(idx);
        if (keep.size() >= MAX_KEEP) break; 

        for (int j = i + 1; j < indices.size(); ++j) {
            int idx_j = indices[j];
            if (suppressed[idx_j]) continue;

            if (IoU(bboxes[idx], bboxes[idx_j]) > iou_threshold) {
                suppressed[idx_j] = true;
            }
        }
    }

    cout << "Number of bounding boxes after NMS: " << keep.size() << endl;
}

int main() {
    int num_boxes = 5;
float boxes[] = {
    100, 100, 210, 210,
    105, 105, 215, 215,
    150, 150, 260, 260,
    200, 200, 310, 310,
    250, 250, 360, 360
};

float scores[] = {0.9, 0.85, 0.8, 0.75, 0.8};
float iou_threshold = 0.5;

std::vector<int> keep_indices;
_nms(boxes, scores, num_boxes, iou_threshold, keep_indices);

for (int i : keep_indices) {
    std::cout << "Keep box index: " << i << std::endl;
    std::cout << "Box coordinates: (" << boxes[i * 4] << ", " << boxes[i * 4 + 1] << ", "
              << boxes[i * 4 + 2] << ", " << boxes[i * 4 + 3] << ")" << std::endl;
    std::cout << "Score: " << scores[i] << std::endl;
}
    return 0;
}