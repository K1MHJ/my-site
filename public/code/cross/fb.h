#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>


using namespace cv;
using namespace std;

struct ImageU8
{
    int width = 0;
    int height = 0;
    std::vector<uint8_t> data;   // row-major: y*width + x

    ImageU8() = default;

    ImageU8(int w, int h, uint8_t value = 0)
        : width(w), height(h), data(static_cast<size_t>(w)* h, value)
    {
    }

    inline uint8_t& at(int x, int y)
    {
        return data[static_cast<size_t>(y) * width + x];
    }

    inline const uint8_t& at(int x, int y) const
    {
        return data[static_cast<size_t>(y) * width + x];
    }
};

struct ImageI32
{
    int width = 0;
    int height = 0;
    std::vector<int32_t> data;

    ImageI32() = default;

    ImageI32(int w, int h, int32_t value = -1)
        : width(w), height(h), data(static_cast<size_t>(w)* h, value)
    {
    }

    inline int32_t& at(int x, int y)
    {
        return data[static_cast<size_t>(y) * width + x];
    }

    inline const int32_t& at(int x, int y) const
    {
        return data[static_cast<size_t>(y) * width + x];
    }
};

struct ImageF32
{
    int width = 0;
    int height = 0;
    std::vector<float> data;

    ImageF32() = default;

    ImageF32(int w, int h, float value = 0.0f)
        : width(w), height(h), data(static_cast<size_t>(w)* h, value)
    {
    }

    inline float& at(int x, int y)
    {
        return data[static_cast<size_t>(y) * width + x];
    }

    inline const float& at(int x, int y) const
    {
        return data[static_cast<size_t>(y) * width + x];
    }
};

// cv::Mat (8bit gray) -> ImageU8
static ImageU8 toImageU8(const cv::Mat& mat)
{
    assert(mat.type() == CV_8UC1);
    ImageU8 img(mat.cols, mat.rows);

    for (int y = 0; y < mat.rows; ++y)
    {
        const uint8_t* srcRow = mat.ptr<uint8_t>(y);
        for (int x = 0; x < mat.cols; ++x)
        {
            img.at(x, y) = srcRow[x];
        }
    }
    return img;
}

// ImageU8 -> cv::Mat (8bit gray)
static cv::Mat toCvMat(const ImageU8& img)
{
    cv::Mat mat(img.height, img.width, CV_8UC1);
    for (int y = 0; y < img.height; ++y)
    {
        uint8_t* dstRow = mat.ptr<uint8_t>(y);
        for (int x = 0; x < img.width; ++x)
        {
            dstRow[x] = img.at(x, y);
        }
    }
    return mat;
}

// 0이 아닌 픽셀 좌표 모으기
struct IntPoint {
    int x;
    int y;

    bool operator==(const IntPoint& other) const noexcept {
        return x == other.x && y == other.y;
    }
    bool operator!=(const IntPoint& other) const noexcept {
        return !(*this == other);
    }
};

struct FloatPoint {
    float x;
    float y;
};
FloatPoint findIntersectionPointXTL(const ImageU8& srcGray,
    int areaThresh = 50,
    int minBranchLen = 60,
    int searchRadius = 10,
    float cornerAngleThresh = 150.0f /*현재는 사용 안 함*/);


// input: 0/255 바이너리, output: 0/255 스켈레톤
void thinningZhangSuen(const ImageU8& srcBin, ImageU8& dstBin);

IntPoint findNearestOnSkeleton(ImageU8& skel, FloatPoint p, int r, bool preferJunction);
std::vector<IntPoint> collectJunctionRegion(const ImageU8& skel, const IntPoint& seed, int maxRadius);
std::vector<std::pair<IntPoint, IntPoint>> findBranchStarts(
    const ImageU8& skel,
    const std::vector<IntPoint>& region);
ImageU8 buildAllowedMask(const ImageU8& skel, const std::vector<IntPoint>& region);

struct LongestPathResult {
    IntPoint endpoint;
    int length;                       // dist(endpoint)
    std::vector<IntPoint> path;      // optional (debug)
};

LongestPathResult followLongestToEndpointBFS(
    const ImageU8& skel,
    const ImageU8& allowed,   // 0/1 mask : allowed==1인 곳만 사용
    const IntPoint& start);