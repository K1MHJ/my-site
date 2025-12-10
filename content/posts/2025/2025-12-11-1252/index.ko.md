---
id: p20251211
slug: p20251211
lang: ko
title: "code"
summary: "첫 만남"
createdAt: 2025-12-11T21:40:00+09:00
updatedAt: 2025-12-11T21:40:00+09:00
tags: ["CODE"]
category: "code"
heroImage: ""
draft: false
---

New
```
// ConsoleApplication1.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#define _USE_MATH_DEFINES // for C++
#include <cmath>

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

// 0이 아닌 픽셀 개수
static int myCountNonZero(const ImageU8& img)
{
    int cnt = 0;
    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            if (img.at(x, y) != 0) cnt++;
        }
    }
    return cnt;
}

// 0이 아닌 픽셀 좌표 모으기
struct IntPoint {
    int x;
    int y;
};

struct FloatPoint {
    float x;
    float y;
};
static void myFindNonZero(const ImageU8& img, std::vector<IntPoint>& pts)
{
    pts.clear();
    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            if (img.at(x, y) != 0)
                pts.push_back({ x, y });
        }
    }
}

// 절댓값 차이 (세선화 수렴 체크용)
static void myAbsDiff(const ImageU8& a, const ImageU8& b, ImageU8& out)
{
    assert(a.width == b.width && a.height == b.height);
    out = ImageU8(a.width, a.height);
    for (int y = 0; y < a.height; ++y)
    {
        for (int x = 0; x < a.width; ++x)
        {
            int d = int(a.at(x, y)) - int(b.at(x, y));
            if (d < 0) d = -d;
            out.at(x, y) = static_cast<uint8_t>(d);
        }
    }
}
static uint8_t computeOtsuThreshold(const ImageU8& gray)
{
    const int histSize = 256;
    int hist[histSize] = { 0 };

    int total = gray.width * gray.height;
    for (int y = 0; y < gray.height; ++y)
    {
        for (int x = 0; x < gray.width; ++x)
        {
            hist[gray.at(x, y)]++;
        }
    }

    double sum = 0.0;
    for (int t = 0; t < histSize; ++t)
        sum += t * hist[t];

    double sumB = 0.0;
    int wB = 0;
    int wF = 0;
    double varMax = 0.0;
    int threshold = 0;

    for (int t = 0; t < histSize; ++t)
    {
        wB += hist[t];
        if (wB == 0) continue;
        wF = total - wB;
        if (wF == 0) break;

        sumB += (double)t * hist[t];
        double mB = sumB / wB;
        double mF = (sum - sumB) / wF;

        double varBetween = (double)wB * (double)wF * (mB - mF) * (mB - mF);
        if (varBetween > varMax)
        {
            varMax = varBetween;
            threshold = t;
        }
    }
    return (uint8_t)threshold;
}

static void myOtsuThreshold(const ImageU8& srcGray,
    ImageU8& dstBin,
    bool invert = true)
{
    uint8_t T = computeOtsuThreshold(srcGray);
    dstBin = ImageU8(srcGray.width, srcGray.height);

    for (int y = 0; y < srcGray.height; ++y)
    {
        for (int x = 0; x < srcGray.width; ++x)
        {
            uint8_t v = (srcGray.at(x, y) > T) ? 255 : 0;
            if (invert) v = 255 - v;
            dstBin.at(x, y) = v;
        }
    }
}
static void myGaussianBlur5x5(const ImageU8& src, ImageU8& dst)
{
    const int k[5] = { 1, 4, 6, 4, 1 };
    const int ksum = 16;

    ImageU8 tmp(src.width, src.height);
    dst = ImageU8(src.width, src.height);

    // 가로
    for (int y = 0; y < src.height; ++y)
    {
        for (int x = 0; x < src.width; ++x)
        {
            int sum = 0;
            for (int i = -2; i <= 2; ++i)
            {
                int xx = x + i;
                if (xx < 0) xx = 0;
                if (xx >= src.width) xx = src.width - 1;
                sum += k[i + 2] * src.at(xx, y);
            }
            tmp.at(x, y) = static_cast<uint8_t>(sum / ksum);
        }
    }

    // 세로
    for (int y = 0; y < src.height; ++y)
    {
        for (int x = 0; x < src.width; ++x)
        {
            int sum = 0;
            for (int j = -2; j <= 2; ++j)
            {
                int yy = y + j;
                if (yy < 0) yy = 0;
                if (yy >= src.height) yy = src.height - 1;
                sum += k[j + 2] * tmp.at(x, yy);
            }
            dst.at(x, y) = static_cast<uint8_t>(sum / ksum);
        }
    }
}
static void myErode3x3(const ImageU8& src, ImageU8& dst)
{
    dst = ImageU8(src.width, src.height);

    for (int y = 0; y < src.height; ++y)
    {
        for (int x = 0; x < src.width; ++x)
        {
            uint8_t m = 255;
            for (int dy = -1; dy <= 1; ++dy)
            {
                int yy = y + dy;
                if (yy < 0) yy = 0;
                if (yy >= src.height) yy = src.height - 1;
                for (int dx = -1; dx <= 1; ++dx)
                {
                    int xx = x + dx;
                    if (xx < 0) xx = 0;
                    if (xx >= src.width) xx = src.width - 1;
                    m = std::min(m, src.at(xx, yy));
                }
            }
            dst.at(x, y) = m;
        }
    }
}

static void myDilate3x3(const ImageU8& src, ImageU8& dst)
{
    dst = ImageU8(src.width, src.height);

    for (int y = 0; y < src.height; ++y)
    {
        for (int x = 0; x < src.width; ++x)
        {
            uint8_t m = 0;
            for (int dy = -1; dy <= 1; ++dy)
            {
                int yy = y + dy;
                if (yy < 0) yy = 0;
                if (yy >= src.height) yy = src.height - 1;
                for (int dx = -1; dx <= 1; ++dx)
                {
                    int xx = x + dx;
                    if (xx < 0) xx = 0;
                    if (xx >= src.width) xx = src.width - 1;
                    m = std::max(m, src.at(xx, yy));
                }
            }
            dst.at(x, y) = m;
        }
    }
}

static void myMorphOpen3x3(const ImageU8& src, ImageU8& dst)
{
    ImageU8 tmp;
    myErode3x3(src, tmp);
    myDilate3x3(tmp, dst);
}

static void myMorphClose3x3(const ImageU8& src, ImageU8& dst)
{
    ImageU8 tmp;
    myDilate3x3(src, tmp);
    myErode3x3(tmp, dst);
}
struct MyComponentStat
{
    int    area = 0;
    double sumX = 0.0;
    double sumY = 0.0;
};

static int myConnectedComponents8(const ImageU8& bin,
    ImageI32& labels,
    std::vector<MyComponentStat>& stats)
{
    const int rows = bin.height;
    const int cols = bin.width;

    labels = ImageI32(cols, rows, -1);
    stats.clear();

    auto inside = [&](int x, int y) {
        return (0 <= x && x < cols && 0 <= y && y < rows);
        };

    const int dx[8] = { -1,0,1,-1,1,-1,0,1 };
    const int dy[8] = { -1,-1,-1,0,0,1,1,1 };

    int curLabel = 0;
    std::vector<IntPoint> q;
    q.reserve(rows * cols / 4);

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            if (bin.at(x, y) == 0) continue;
            if (labels.at(x, y) != -1) continue;

            stats.push_back(MyComponentStat{});
            MyComponentStat& st = stats.back();
            q.clear();
            q.push_back({ x, y });
            labels.at(x, y) = curLabel;

            for (size_t qi = 0; qi < q.size(); ++qi)
            {
                IntPoint p = q[qi];
                st.area++;
                st.sumX += p.x;
                st.sumY += p.y;

                for (int k = 0; k < 8; ++k)
                {
                    int nx = p.x + dx[k];
                    int ny = p.y + dy[k];
                    if (!inside(nx, ny)) continue;
                    if (bin.at(nx, ny) == 0) continue;
                    int32_t& lab = labels.at(nx, ny);
                    if (lab != -1) continue;
                    lab = curLabel;
                    q.push_back({ nx, ny });
                }
            }
            curLabel++;
        }
    }
    return curLabel;
}

// 작은 blob 제거
static void removeSmallComponents(ImageU8& bin,
    const ImageI32& labels,
    const std::vector<MyComponentStat>& stats,
    int areaThresh)
{
    for (int y = 0; y < bin.height; ++y)
    {
        for (int x = 0; x < bin.width; ++x)
        {
            int lab = labels.at(x, y);
            if (lab < 0) continue;
            if (stats[lab].area < areaThresh)
                bin.at(x, y) = 0;
        }
    }
}
static void myDistanceTransformL2Approx(const ImageU8& bin, ImageF32& dist)
{
    const int rows = bin.height;
    const int cols = bin.width;

    dist = ImageF32(cols, rows);
    const float INF = 1e9f;

    // 초기화: 선(전경)=0, 배경=INF
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            dist.at(x, y) = (bin.at(x, y) != 0) ? 0.0f : INF;
        }
    }

    const float w1 = 1.0f;
    const float w2 = 1.41421356f; // sqrt(2)

    // forward pass
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            float v = dist.at(x, y);
            if (v == 0.0f) continue;

            for (int dy = -1; dy <= 0; ++dy)
            {
                int yy = y + dy;
                if (yy < 0) continue;
                for (int dx = -1; dx <= 1; ++dx)
                {
                    int xx = x + dx;
                    if (xx < 0 || xx >= cols) continue;
                    if (dy == 0 && dx >= 0) continue;

                    float cost = (dx == 0 || dy == 0) ? w1 : w2;
                    float cand = dist.at(xx, yy) + cost;
                    if (cand < v) v = cand;
                }
            }
            dist.at(x, y) = v;
        }
    }

    // backward pass
    for (int y = rows - 1; y >= 0; --y)
    {
        for (int x = cols - 1; x >= 0; --x)
        {
            float v = dist.at(x, y);
            if (v == 0.0f) continue;

            for (int dy = 0; dy <= 1; ++dy)
            {
                int yy = y + dy;
                if (yy >= rows) continue;
                for (int dx = -1; dx <= 1; ++dx)
                {
                    int xx = x + dx;
                    if (xx < 0 || xx >= cols) continue;
                    if (dy == 0 && dx <= 0) continue;

                    float cost = (dx == 0 || dy == 0) ? w1 : w2;
                    float cand = dist.at(xx, yy) + cost;
                    if (cand < v) v = cand;
                }
            }
            dist.at(x, y) = v;
        }
    }
}
// 8이웃 흰색(1) 픽셀 개수
static int countWhiteNeighbors(const ImageU8& img, int x, int y)
{
    int cnt = 0;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0) continue;
            int xx = x + dx;
            int yy = y + dy;
            if (xx < 0 || xx >= img.width || yy < 0 || yy >= img.height) continue;
            if (img.at(xx, yy) != 0) cnt++;
        }
    }
    return cnt;
}

// 0->1 전이 횟수 (시계 순서 p2..p9..p2)
static int countTransitions(const ImageU8& img, int x, int y)
{
    int px[8][2] = {
        {0,-1}, {1,-1}, {1,0}, {1,1},
        {0,1}, {-1,1}, {-1,0}, {-1,-1}
    };

    int A = 0;
    for (int k = 0; k < 8; ++k)
    {
        int x1 = x + px[k][0];
        int y1 = y + px[k][1];
        int x2 = x + px[(k + 1) % 8][0];
        int y2 = y + px[(k + 1) % 8][1];

        uint8_t p1 = 0, p2 = 0;
        if (x1 >= 0 && x1 < img.width && y1 >= 0 && y1 < img.height)
            p1 = img.at(x1, y1);
        if (x2 >= 0 && x2 < img.width && y2 >= 0 && y2 < img.height)
            p2 = img.at(x2, y2);

        if (p1 == 0 && p2 == 1) A++;
    }
    return A;
}

static void thinningIteration(ImageU8& im, int iter)
{
    std::vector<uint8_t> marker(im.width * im.height, 0);

    auto get = [&](int x, int y) -> uint8_t {
        return im.at(x, y);
        };

    for (int y = 1; y < im.height - 1; ++y)
    {
        for (int x = 1; x < im.width - 1; ++x)
        {
            if (im.at(x, y) != 1) continue;

            int B = countWhiteNeighbors(im, x, y);
            if (B < 2 || B > 6) continue;

            int A = countTransitions(im, x, y);
            if (A != 1) continue;

            uint8_t p2 = get(x, y - 1);
            uint8_t p4 = get(x + 1, y);
            uint8_t p6 = get(x, y + 1);
            uint8_t p8 = get(x - 1, y);

            if (iter == 0)
            {
                if ((p2 * p4 * p6) == 0 &&
                    (p4 * p6 * p8) == 0)
                {
                    marker[y * im.width + x] = 1;
                }
            }
            else
            {
                if ((p2 * p4 * p8) == 0 &&
                    (p2 * p6 * p8) == 0)
                {
                    marker[y * im.width + x] = 1;
                }
            }
        }
    }

    for (int y = 0; y < im.height; ++y)
    {
        for (int x = 0; x < im.width; ++x)
        {
            if (marker[y * im.width + x])
                im.at(x, y) = 0;
        }
    }
}

static bool myImageEqual(const ImageU8& a, const ImageU8& b)
{
    if (a.width != b.width || a.height != b.height) return false;
    return a.data == b.data;
}

// input: 0/255 바이너리, output: 0/255 스켈레톤
static void thinningZhangSuen(const ImageU8& srcBin, ImageU8& dstBin)
{
    assert(srcBin.width > 0 && srcBin.height > 0);
    dstBin = ImageU8(srcBin.width, srcBin.height);

    // 0/255 → 0/1
    for (int y = 0; y < srcBin.height; ++y)
        for (int x = 0; x < srcBin.width; ++x)
            dstBin.at(x, y) = (srcBin.at(x, y) ? 1 : 0);

    ImageU8 prev(dstBin.width, dstBin.height);

    while (true)
    {
        prev = dstBin;
        thinningIteration(dstBin, 0);
        thinningIteration(dstBin, 1);
        if (myImageEqual(dstBin, prev)) break;
    }

    // 0/1 → 0/255
    for (int y = 0; y < dstBin.height; ++y)
        for (int x = 0; x < dstBin.width; ++x)
            dstBin.at(x, y) = dstBin.at(x, y) ? 255 : 0;
}

static bool inBounds(const ImageU8& img, int x, int y)
{
    return (0 <= x && x < img.width &&
        0 <= y && y < img.height);
}

// 8-이웃 중 흰 픽셀(0이 아닌 값) 개수
static int countNeighbors(const ImageU8& skel, int x, int y)
{
    int cnt = 0;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0) continue;
            int xx = x + dx;
            int yy = y + dy;
            if (!inBounds(skel, xx, yy)) continue;
            if (skel.at(xx, yy) != 0) cnt++;
        }
    }
    return cnt;
}

// 8-이웃 흰 픽셀 좌표 리스트
static void getNeighbors(const ImageU8& skel,
    int x, int y,
    std::vector<IntPoint>& neighbors)
{
    neighbors.clear();
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0) continue;
            int xx = x + dx;
            int yy = y + dy;
            if (!inBounds(skel, xx, yy)) continue;
            if (skel.at(xx, yy) != 0)
                neighbors.push_back({ xx, yy });
        }
    }
}
//-------------------------------------------
// skeleton에서 (cur, prev)를 기준으로 다음 픽셀 찾기 (ImageU8 버전)
//-------------------------------------------
static bool findNextSkeletonPixel(const ImageU8& skel,
    const IntPoint& cur,
    const IntPoint& prev,
    IntPoint& next)
{
    for (int dy = -1; dy <= 1; ++dy)
    {
        int yy = cur.y + dy;
        if (yy < 0 || yy >= skel.height) continue;

        for (int dx = -1; dx <= 1; ++dx)
        {
            int xx = cur.x + dx;
            if (dx == 0 && dy == 0) continue;
            if (xx < 0 || xx >= skel.width) continue;
            if (skel.at(xx, yy) == 0) continue;
            if (xx == prev.x && yy == prev.y) continue;

            next = { xx, yy };
            return true;
        }
    }
    return false;
}
// 한 방향 가지 길이 측정 (시작점 center, 이웃 firstNeighbor에서 출발)
static int measureBranchLength(const ImageU8& skel,
    const IntPoint& center,
    const IntPoint& firstNeighbor,
    int maxSteps = 1000)
{
    IntPoint prev = center;
    IntPoint cur = firstNeighbor;
    int len = 1; // firstNeighbor까지 1

    for (int step = 0; step < maxSteps; ++step)
    {
        int deg = countNeighbors(skel, cur.x, cur.y);

        // 현재가 교차점(deg>=3) 또는 끝점(deg==1)이면 여기까지
        if (deg != 2)
            break;

        IntPoint next;
        if (!findNextSkeletonPixel(skel, cur, prev, next))
            break;

        prev = cur;
        cur = next;
        ++len;
    }
    return len;
}
//static bool inBounds(const ImageU8& img, int x, int y)
//{
//    return (0 <= x && x < img.width &&
//        0 <= y && y < img.height);
//}
//
//// 8-이웃 중 흰 픽셀(0이 아닌 값) 개수
//static int countWhiteNeighbors(const ImageU8& skel, int x, int y)
//{
//    int cnt = 0;
//    for (int dy = -1; dy <= 1; ++dy)
//    {
//        for (int dx = -1; dx <= 1; ++dx)
//        {
//            if (dx == 0 && dy == 0) continue;
//            int xx = x + dx;
//            int yy = y + dy;
//            if (!inBounds(skel, xx, yy)) continue;
//            if (skel.at(xx, yy) != 0) cnt++;
//        }
//    }
//    return cnt;
//}

//// 8-이웃 흰 픽셀 좌표
//static void getNeighbors(const ImageU8& skel,
//    int x, int y,
//    std::vector<IntPoint>& neighbors)
//{
//    neighbors.clear();
//    for (int dy = -1; dy <= 1; ++dy)
//    {
//        for (int dx = -1; dx <= 1; ++dx)
//        {
//            if (dx == 0 && dy == 0) continue;
//            int xx = x + dx;
//            int yy = y + dy;
//            if (!inBounds(skel, xx, yy)) continue;
//            if (skel.at(xx, yy) != 0)
//                neighbors.push_back({ xx, yy });
//        }
//    }
//}

//
////-------------------------------------------
//// skeleton에서 (cur, prev)를 기준으로 다음 픽셀 찾기 (ImageU8 버전)
////-------------------------------------------
//static bool findNextSkeletonPixel(const ImageU8& skel,
//    const IntPoint& cur,
//    const IntPoint& prev,
//    IntPoint& next)
//{
//    for (int dy = -1; dy <= 1; ++dy)
//    {
//        int yy = cur.y + dy;
//        if (yy < 0 || yy >= skel.height) continue;
//
//        for (int dx = -1; dx <= 1; ++dx)
//        {
//            int xx = cur.x + dx;
//            if (dx == 0 && dy == 0) continue;
//            if (xx < 0 || xx >= skel.width) continue;
//            if (skel.at(xx, yy) == 0) continue;
//            if (xx == prev.x && yy == prev.y) continue;
//
//            next = { xx, yy };
//            return true;
//        }
//    }
//    return false;
//}
// center -> firstNeighbor 방향으로 가지를 따라가며
// 길이(len)와 끝점(end)을 반환
static int followBranch(const ImageU8& skel,
    const IntPoint& center,
    const IntPoint& firstNeighbor,
    IntPoint& end,
    int maxSteps = 1000)
{
    IntPoint prev = center;
    IntPoint cur = firstNeighbor;

    int len = 1; // firstNeighbor까지 1

    for (int step = 0; step < maxSteps; ++step)
    {
        int deg = countWhiteNeighbors(skel, cur.x, cur.y);

        // deg!=2 이면 끝점(1) 혹은 교차점(>=3)이므로 종료
        if (deg != 2)
            break;

        IntPoint next;
        if (!findNextSkeletonPixel(skel, cur, prev, next))
            break;

        prev = cur;
        cur = next;
        ++len;
    }

    end = cur;
    return len;
}
// skel: 0/255 스켈레톤
// minBranchLen: 교차점에서 나온 가지 중 이 길이보다 짧으면 삭제
static void pruneShortBranches(ImageU8& skel, int minBranchLen)
{
    const int W = skel.width;
    const int H = skel.height;

    ImageU8 toDelete(W, H, 0);

    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            if (skel.at(x, y) == 0) continue;

            int deg = countWhiteNeighbors(skel, x, y);
            if (deg != 1) continue; // 끝점만 시작점

            IntPoint start{ x, y };
            IntPoint prev = start;
            IntPoint cur = start;

            std::vector<IntPoint> path;
            path.push_back(cur);

            int len = 0;

            while (true)
            {
                IntPoint next;
                if (!findNextSkeletonPixel(skel, cur, prev, next))
                    break;

                prev = cur;
                cur = next;
                path.push_back(cur);
                ++len;

                int d = countWhiteNeighbors(skel, cur.x, cur.y);
                if (d != 2) // 교차점 또는 끝점
                    break;
            }

            int endDeg = countWhiteNeighbors(skel, cur.x, cur.y);

            // 끝점 -> 교차점 가지이고, 길이가 짧으면 삭제
            if (endDeg >= 3 && len < minBranchLen)
            {
                // 마지막 점(cur)은 교차점이므로 유지, 나머지 path 삭제
                for (size_t i = 0; i + 1 < path.size(); ++i)
                {
                    const auto& p = path[i];
                    toDelete.at(p.x, p.y) = 255;
                }
            }
        }
    }

    // 실제 삭제 적용
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            if (toDelete.at(x, y) != 0)
                skel.at(x, y) = 0;
        }
    }
}
static bool isXorTIntersectionCandidate(const ImageU8& skel,
    int x, int y,
    float minBranchLenXT,
    float angleMinXT,
    float angleMaxXT)
{
    std::vector<IntPoint> neighbors;
    getNeighbors(skel, x, y, neighbors);
    if (neighbors.size() < 3) return false;

    const IntPoint center{ x, y };

    struct BranchDir {
        float vx, vy;  // 방향 단위벡터
        float len;
    };

    std::vector<BranchDir> dirs;

    for (const auto& nb : neighbors)
    {
        IntPoint end;
        int len = followBranch(skel, center, nb, end, /*maxSteps=*/1000);
        if (len < (int)minBranchLenXT) continue;

        float dx = float(end.x - center.x);
        float dy = float(end.y - center.y);
        float L = std::sqrt(dx * dx + dy * dy);
        if (L <= 0.0f) continue;

        dirs.push_back({ dx / L, dy / L, (float)len });
    }

    if (dirs.size() < 3) return false;

    // 가지 방향 사이 각도 검사
    for (size_t i = 0; i < dirs.size(); ++i)
    {
        for (size_t j = i + 1; j < dirs.size(); ++j)
        {
            float dot = dirs[i].vx * dirs[j].vx + dirs[i].vy * dirs[j].vy;
            // 수치 오차 보정
            dot = std::max(-1.0f, std::min(1.0f, dot));

            float angleDeg = std::acos(dot) * 180.0f / float(M_PI);
            if (angleDeg >= angleMinXT && angleDeg <= angleMaxXT)
            {
                // 최소 한 쌍의 가지가 원하는 각도 범위에 있으면 X/T 후보로 인정
                return true;
            }
        }
    }

    return false;
}
static bool getCornerAngleDegGlobal(const ImageU8& skel,
    int x, int y,
    float& outAngleDeg,
    int maxStep = 10)
{
    std::vector<IntPoint> neighbors;
    getNeighbors(skel, x, y, neighbors);

    if (neighbors.size() != 2)
        return false;

    IntPoint center{ x, y };

    IntPoint end1, end2;
    int len1 = followBranch(skel, center, neighbors[0], end1, maxStep);
    int len2 = followBranch(skel, center, neighbors[1], end2, maxStep);

    if (len1 < 1 || len2 < 1)
        return false;

    float dx1 = float(end1.x - center.x);
    float dy1 = float(end1.y - center.y);
    float dx2 = float(end2.x - center.x);
    float dy2 = float(end2.y - center.y);

    float L1 = std::sqrt(dx1 * dx1 + dy1 * dy1);
    float L2 = std::sqrt(dx2 * dx2 + dy2 * dy2);
    if (L1 <= 0.0f || L2 <= 0.0f)
        return false;

    dx1 /= L1; dy1 /= L1;
    dx2 /= L2; dy2 /= L2;

    float dot = dx1 * dx2 + dy1 * dy2;
    dot = std::max(-1.0f, std::min(1.0f, dot));

    outAngleDeg = std::acos(dot) * 180.0f / float(M_PI);
    return true;
}
static FloatPoint refineCenterSubpixel(const ImageF32& dist,
    const IntPoint& pt)
{
    float sumW = 0.0f;
    float sumX = 0.0f;
    float sumY = 0.0f;

    for (int dy = -1; dy <= 1; ++dy)
    {
        int yy = pt.y + dy;
        if (yy < 0 || yy >= dist.height) continue;

        for (int dx = -1; dx <= 1; ++dx)
        {
            int xx = pt.x + dx;
            if (xx < 0 || xx >= dist.width) continue;

            float w = dist.at(xx, yy);
            if (w <= 0.0f) continue;

            sumW += w;
            sumX += w * float(xx);
            sumY += w * float(yy);
        }
    }

    FloatPoint result;
    if (sumW > 0.0f)
    {
        result.x = sumX / sumW;
        result.y = sumY / sumW;
    }
    else
    {
        result.x = (float)pt.x;
        result.y = (float)pt.y;
    }
    return result;
}
//-------------------------------------------
// X/T/L 교차/코너점 찾기 (노이즈/굵기 고려, ImageU8 버전)
//-------------------------------------------
static FloatPoint findIntersectionPointXTL(const ImageU8& srcGray,
    int areaThresh = 50,
    int minBranchLen = 60,
    int searchRadius = 10,
    float cornerAngleThresh = 150.0f /*현재는 사용 안 함*/)
{
    // 1) 이진화 (선 = 255, 배경 = 0)
    ImageU8 blur;
    myGaussianBlur5x5(srcGray, blur);

    ImageU8 bin;
    myOtsuThreshold(blur, bin, /*invert=*/true);

    // 모폴로지로 노이즈 정리
    ImageU8 tmp;
    myMorphOpen3x3(bin, tmp);
    myMorphClose3x3(tmp, bin);

    // 작은 blob 제거
    ImageI32 labels;
    std::vector<MyComponentStat> stats;
    int nLabels = myConnectedComponents8(bin, labels, stats);
    removeSmallComponents(bin, labels, stats, areaThresh);

    // 2) 세선화
    ImageU8 skel;
    thinningZhangSuen(bin, skel);

    // 3) 가지 pruning
    pruneShortBranches(skel, minBranchLen);

    

    // 4) X/T junction + L corner 후보 마스크 생성
    ImageU8 candMask(skel.width, skel.height, 0);

    for (int y = 1; y < skel.height - 1; ++y)
    {
        for (int x = 1; x < skel.width - 1; ++x)
        {
            if (skel.at(x, y) == 0) continue;

            int deg = countWhiteNeighbors(skel, x, y);

            // --- X / T 후보 (deg >= 3) ---
            if (deg >= 3)
            {
                float minBranchLenXT = 10.0f;   // 해상도에 맞게 조정
                float angleMinXT = 60.0f;
                float angleMaxXT = 120.0f;

                if (isXorTIntersectionCandidate(skel, x, y,
                    minBranchLenXT,
                    angleMinXT,
                    angleMaxXT))
                {
                    candMask.at(x, y) = 255;
                }
            }
            // --- L 코너 후보 (deg == 2) ---
            else if (deg == 2)
            {
                float angleDeg;
                if (getCornerAngleDegGlobal(skel, x, y, angleDeg, /*maxStep=*/10))
                {
                    // L은 60~120도, 직선(~180도)은 제외
                    if (angleDeg > 60.0f && angleDeg < 120.0f)
                    {
                        candMask.at(x, y) = 255;
                    }
                }
            }
        }
    }
    // 3) 결과를 다시 Mat으로 변환해서 화면 표시
    cv::Mat skelMat = toCvMat(candMask);
    cv::imshow("skel", skelMat);
    cv::waitKey(0);

    if (myCountNonZero(candMask) == 0)
    {
        // 교차/코너 못 찾은 경우
        return FloatPoint{ 0.0f, 0.0f };
    }

    // 4-1) 가장 큰 candidate 컴포넌트 선택 (교차/코너가 1개라는 가정)
    ImageI32 cLabels;
    std::vector<MyComponentStat> cStats;
    int cNum = myConnectedComponents8(candMask, cLabels, cStats);

    int bestLabel = -1;
    int bestArea = 0;
    for (int i = 0; i < cNum; ++i)
    {
        int area = cStats[i].area;
        if (area > bestArea)
        {
            bestArea = area;
            bestLabel = i;
        }
    }

    FloatPoint approx{ 0.0f, 0.0f };

    if (bestLabel < 0)
    {
        // 이론상 여기 오기 어렵지만, 방어 코드
        std::vector<IntPoint> pts;
        myFindNonZero(candMask, pts);
        for (auto& p : pts)
        {
            approx.x += (float)p.x;
            approx.y += (float)p.y;
        }
        approx.x /= (float)pts.size();
        approx.y /= (float)pts.size();
    }
    else
    {
        const auto& st = cStats[bestLabel];
        approx.x = (float)(st.sumX / st.area);
        approx.y = (float)(st.sumY / st.area);
    }

    // 5) distance transform + sub-pixel 중앙 보정
    ImageF32 dist;
    myDistanceTransformL2Approx(bin, dist);

    IntPoint bestPt{
        (int)std::lround(approx.x),
        (int)std::lround(approx.y)
    };

    bestPt.x = std::clamp(bestPt.x, 0, dist.width - 1);
    bestPt.y = std::clamp(bestPt.y, 0, dist.height - 1);

    float bestVal = dist.at(bestPt.x, bestPt.y);

    for (int dy = -searchRadius; dy <= searchRadius; ++dy)
    {
        int yy = bestPt.y + dy;
        if (yy < 0 || yy >= dist.height) continue;

        for (int dx = -searchRadius; dx <= searchRadius; ++dx)
        {
            int xx = bestPt.x + dx;
            if (xx < 0 || xx >= dist.width) continue;

            float v = dist.at(xx, yy);
            if (v > bestVal)
            {
                bestVal = v;
                bestPt.x = xx;
                bestPt.y = yy;
            }
        }
    }

    FloatPoint refined = refineCenterSubpixel(dist, bestPt);
    return refined;
}
//
//// 짧은 가지를 잘라내는 함수
//// skel: 0/255 스켈레톤 이미지 (0=배경, 255=선)
//// minBranchLen: 이 길이보다 짧고, 교차점에서 나온 가지는 삭제
//static void pruneShortBranches(ImageU8& skel, int minBranchLen)
//{
//    const int W = skel.width;
//    const int H = skel.height;
//
//    // 삭제 마스크 (나중에 한 번에 지우기)
//    ImageU8 toDelete(W, H, 0);
//
//    // 모든 픽셀 스캔
//    for (int y = 0; y < H; ++y)
//    {
//        for (int x = 0; x < W; ++x)
//        {
//            if (skel.at(x, y) == 0) continue;
//
//            int deg = countNeighbors(skel, x, y);
//            if (deg != 1) continue; // 끝점만 시작점 후보
//
//            IntPoint start{ x, y };
//            IntPoint prev{ x, y };
//            IntPoint cur{ x, y };
//
//            std::vector<IntPoint> path;
//            path.push_back(cur);
//
//            int len = 0;
//
//            // 끝점에서 시작해서 교차점/또다른 끝점까지 추적
//            while (true)
//            {
//                IntPoint next;
//                if (!findNextSkeletonPixel(skel, cur, prev, next))
//                    break; // 더 갈 곳 없음
//
//                prev = cur;
//                cur = next;
//                path.push_back(cur);
//                ++len;
//
//                int d = countNeighbors(skel, cur.x, cur.y);
//                if (d != 2) // 교차점(deg>=3) 또는 끝점(deg==1)
//                    break;
//            }
//
//            int endDeg = countNeighbors(skel, cur.x, cur.y);
//
//            // 1) 끝점→교차점으로 이어지는 가지이고
//            // 2) 길이가 짧으면 삭제
//            if (endDeg >= 3 && len < minBranchLen)
//            {
//                // 마지막 점(cur)은 교차점이므로 남기고, 나머지만 삭제
//                for (size_t i = 0; i + 1 < path.size(); ++i)
//                {
//                    const auto& p = path[i];
//                    toDelete.at(p.x, p.y) = 255;
//                }
//            }
//            // 끝점→끝점으로 이어지는 것 (직선 한 줄)은 삭제하지 않는다.
//        }
//    }
//
//    // 실제로 삭제 적용
//    for (int y = 0; y < H; ++y)
//    {
//        for (int x = 0; x < W; ++x)
//        {
//            if (toDelete.at(x, y) != 0)
//                skel.at(x, y) = 0;
//        }
//    }
//}

int main() {
    cv::Mat src = cv::imread("D:\\Devs\\ClockImageReader\\clockimages\\cut-+.jpg", cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cout << "이미지를 불러올 수 없습니다.\n";
        return -1;
    }

    if (src.channels() == 3)
        cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
    else
        src = src.clone();


    // 1) cv::Mat -> ImageU8
    ImageU8 img = toImageU8(src);


    // 2) 교차점 찾기
    std::vector<IntPoint> intersections;
    FloatPoint intersection = findIntersectionPointXTL(img);

    // 이제 intersections 안에 교차점 좌표들이 들어 있음

    // 3) 결과를 다시 Mat으로 변환해서 화면 표시
    cv::Mat skelMat = toCvMat(img);
    cv::imshow("skel", skelMat);
    cv::waitKey(0);



    Mat color;
    cvtColor(src, color, COLOR_GRAY2BGR);
    //for (auto is : intersections) 
    {
        circle(color, { (int)intersection.x, (int)intersection.y }, 5, Scalar(0, 0, 255), -1);
    }

    imshow("result", color);
    waitKey(0);

    return 0;
}
```

Classic
```
// ConsoleApplication1.cpp : ?? ??????? 'main' ????? ???????. ??? ???��?? ?????? ?????? ???????.
//
#include <windows.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <cstddef>
#include <array>
#include <vector>
#include <queue>
#include <cstdint>
#include <utility>
#include <algorithm>

using namespace cv;
using namespace std;

static int countWhiteNeighbors(const Mat& img, int x, int y);
static bool findNextSkeletonPixel(const Mat& skel,
    const Point& cur,
    const Point& prev,
    Point& next);

struct BranchInfo
{
    cv::Point2f dir; // ???? ???? ????
    float       length; // ??? ??
};

// (center -> firstNeighbor) ???????? ?????? ????? ????/????? ?????.
static BranchInfo traceBranch(const cv::Mat& skel,
    const cv::Point& center,
    const cv::Point& firstNeighbor,
    int maxStep = 30)
{
    BranchInfo info;
    info.dir = cv::Point2f(0.f, 0.f);
    info.length = 0.f;

    cv::Point prev = center;
    cv::Point cur = firstNeighbor;

    std::vector<cv::Point> pts;
    pts.push_back(center);
    pts.push_back(cur);

    for (int step = 0; step < maxStep; ++step)
    {
        cv::Point next;
        if (!findNextSkeletonPixel(skel, cur, prev, next))
            break;

        pts.push_back(next);
        prev = cur;
        cur = next;

        int deg = countWhiteNeighbors(skel, cur.x, cur.y);
        if (deg != 2) // ???? or ?��??????? ????
            break;
    }

    if (pts.size() < 2) return info;

    cv::Point2f v((float)(pts.back().x - center.x),
        (float)(pts.back().y - center.y));
    float n = std::sqrt(v.x * v.x + v.y * v.y);
    if (n < 1e-3f) return info;

    info.dir.x = v.x / n;
    info.dir.y = v.y / n;
    info.length = n; // center ???? ???

    return info;
}

static float angleBetweenDeg(const cv::Point2f& a, const cv::Point2f& b)
{
    float n1 = std::sqrt(a.x * a.x + a.y * a.y);
    float n2 = std::sqrt(b.x * b.x + b.y * b.y);
    if (n1 < 1e-3f || n2 < 1e-3f) return 180.f;

    float dot = (a.x * b.x + a.y * b.y) / (n1 * n2);
    dot = std::max(-1.0f, std::min(1.0f, dot));
    float theta = std::acos(dot) * 180.0f / (float)CV_PI;
    return theta;
}
// X/T ??? ????: ????? ?? ???? 2?? ??? + ???? ????? ?
static bool isXorTIntersectionCandidate(const cv::Mat& skel,
    int cx, int cy,
    float minBranchLenXT,
    float angleMinXT,
    float angleMaxXT)
{
    if (skel.at<uchar>(cy, cx) == 0) return false;

    // ??? ???
    std::vector<cv::Point> neigh;
    for (int dy = -1; dy <= 1; ++dy)
    {
        int yy = cy + dy;
        if (yy < 0 || yy >= skel.rows) continue;
        const uchar* row = skel.ptr<uchar>(yy);
        for (int dx = -1; dx <= 1; ++dx)
        {
            int xx = cx + dx;
            if (dx == 0 && dy == 0) continue;
            if (xx < 0 || xx >= skel.cols) continue;
            if (row[xx] == 0) continue;
            neigh.emplace_back(xx, yy);
        }
    }
    if (neigh.size() < 3) return false; // deg < 3

    // ?? ??? ???????? ???? ????
    std::vector<BranchInfo> branches;
    branches.reserve(neigh.size());
    for (auto& nb : neigh)
    {
        BranchInfo bi = traceBranch(skel, cv::Point(cx, cy), nb, /*maxStep=*/30);
        if (bi.length >= minBranchLenXT)
            branches.push_back(bi);
    }

    if (branches.size() < 2)
        return false; // ????? ?? ?????? 2?? ?????? ??????? ????

    // ???? ?? ?? ???? ???
    std::sort(branches.begin(), branches.end(),
        [](const BranchInfo& a, const BranchInfo& b) {
            return a.length > b.length;
        });

    const cv::Point2f& d1 = branches[0].dir;
    const cv::Point2f& d2 = branches[1].dir;

    float ang = angleBetweenDeg(d1, d2);

    // X/T?? ???, ?????? ???? ???? ????? ???? ??? 60~120?? ?????? ???
    if (ang >= angleMinXT && ang <= angleMaxXT)
        return true;

    return false;
}

// deg == 2 ?? ??????? ????? "????????" ???? ???? ???
// ??????? ???? ??? ????? ?? ?????? ???? ?????? ????
static bool getCornerAngleDegGlobal(const cv::Mat& skel,
    int x, int y,
    float& outDeg,
    int maxStep = 10)
{
    if (skel.at<uchar>(y, x) == 0) return false;

    // 1) ??? ?? ?? ???
    std::vector<cv::Point> neigh;
    for (int dy = -1; dy <= 1; ++dy)
    {
        int yy = y + dy;
        if (yy < 0 || yy >= skel.rows) continue;
        const uchar* row = skel.ptr<uchar>(yy);
        for (int dx = -1; dx <= 1; ++dx)
        {
            int xx = x + dx;
            if (dx == 0 && dy == 0) continue;
            if (xx < 0 || xx >= skel.cols) continue;
            if (row[xx] == 0) continue;
            neigh.emplace_back(xx, yy);
        }
    }
    if (neigh.size() != 2) return false; // deg != 2 ??? ??? ???

    auto traceDir = [&](const cv::Point& first,
        const cv::Point& from,
        std::vector<cv::Point>& pts)
        {
            pts.clear();
            cv::Point cur = first;
            cv::Point prev = from;
            pts.push_back(cur);

            for (int step = 0; step < maxStep; ++step)
            {
                cv::Point next;
                if (!findNextSkeletonPixel(skel, cur, prev, next))
                    break;

                pts.push_back(next);
                prev = cur;
                cur = next;

                int deg = countWhiteNeighbors(skel, cur.x, cur.y);
                if (deg != 2) break; // ???? ?��???/???????? ????
            }
        };

    // 2) ?? ???????? ?????? ???? ??, ?/?????? ?????? ???? ???? ????
    std::vector<cv::Point> pts1, pts2;
    traceDir(neigh[0], cv::Point(x, y), pts1);
    traceDir(neigh[1], cv::Point(x, y), pts2);

    if (pts1.size() < 2 || pts2.size() < 2) return false;

    auto dirFromPoints = [](const std::vector<cv::Point>& pts)->cv::Point2f
        {
            cv::Point2f v((float)(pts.back().x - pts.front().x),
                (float)(pts.back().y - pts.front().y));
            float n = std::sqrt(v.x * v.x + v.y * v.y);
            if (n < 1e-3f) return cv::Point2f(0, 0);
            v.x /= n; v.y /= n;
            return v;
        };

    cv::Point2f v1 = dirFromPoints(pts1);
    cv::Point2f v2 = dirFromPoints(pts2);
    float n1 = cv::sqrt(v1.x * v1.x + v1.y * v1.y);
    float n2 = cv::sqrt(v2.x * v2.x + v2.y * v2.y);
    if (n1 < 1e-3f || n2 < 1e-3f) return false;

    // dot = cos(theta)
    float cosTheta = v1.x * v2.x + v1.y * v2.y;
    cosTheta = std::max(-1.0f, std::min(1.0f, cosTheta));
    float theta = std::acos(cosTheta) * 180.0f / (float)CV_PI; // deg

    outDeg = theta;
    return true;
}

//-------------------------------------------
// Zhang-Suen thinning
//-------------------------------------------

static void thinningIteration(cv::Mat& im, int iter)
{
    CV_Assert(im.type() == CV_8UC1);
    Mat marker = Mat::zeros(im.size(), CV_8UC1);

    int rows = im.rows;
    int cols = im.cols;

    for (int y = 1; y < rows - 1; ++y)
    {
        uchar* pRow = im.ptr<uchar>(y);
        uchar* pRowUp = im.ptr<uchar>(y - 1);
        uchar* pRowDn = im.ptr<uchar>(y + 1);

        for (int x = 1; x < cols - 1; ++x)
        {
            uchar p1 = pRow[x];
            if (p1 == 0) continue;

            // 8 neighbors (p2 ~ p9)
            uchar p2 = pRowUp[x];
            uchar p3 = pRowUp[x + 1];
            uchar p4 = pRow[x + 1];
            uchar p5 = pRowDn[x + 1];
            uchar p6 = pRowDn[x];
            uchar p7 = pRowDn[x - 1];
            uchar p8 = pRow[x - 1];
            uchar p9 = pRowUp[x - 1];

            int A = (p2 == 0 && p3 > 0) + (p3 == 0 && p4 > 0) +
                (p4 == 0 && p5 > 0) + (p5 == 0 && p6 > 0) +
                (p6 == 0 && p7 > 0) + (p7 == 0 && p8 > 0) +
                (p8 == 0 && p9 > 0) + (p9 == 0 && p2 > 0);

            int B = (p2 > 0) + (p3 > 0) + (p4 > 0) + (p5 > 0) +
                (p6 > 0) + (p7 > 0) + (p8 > 0) + (p9 > 0);

            if (A == 1 && (B >= 2 && B <= 6))
            {
                if (iter == 0)
                {
                    if ((p2 * p4 * p6) == 0 &&
                        (p4 * p6 * p8) == 0)
                    {
                        marker.at<uchar>(y, x) = 1;
                    }
                }
                else
                {
                    if ((p2 * p4 * p8) == 0 &&
                        (p2 * p6 * p8) == 0)
                    {
                        marker.at<uchar>(y, x) = 1;
                    }
                }
            }
        }
    }

    im &= ~marker;
}

static void thinningZhangSuen(const cv::Mat& src, cv::Mat& dst)
{
    CV_Assert(src.type() == CV_8UC1);
    dst = src.clone();
    dst /= 255;  // 0/1 ??
    Mat prev = Mat::zeros(dst.size(), CV_8UC1);
    Mat diff;

    do {
        thinningIteration(dst, 0);
        thinningIteration(dst, 1);
        absdiff(dst, prev, diff);
        dst.copyTo(prev);
    } while (countNonZero(diff) > 0);

    dst *= 255;  // ??? 0/255
}

//-------------------------------------------
// 8-??? ?? ??? ????
//-------------------------------------------
static int countWhiteNeighbors(const Mat& img, int x, int y)
{
    int cnt = 0;
    for (int dy = -1; dy <= 1; ++dy)
    {
        int yy = y + dy;
        if (yy < 0 || yy >= img.rows) continue;
        const uchar* row = img.ptr<uchar>(yy);
        for (int dx = -1; dx <= 1; ++dx)
        {
            int xx = x + dx;
            if (dx == 0 && dy == 0) continue;
            if (xx < 0 || xx >= img.cols) continue;
            if (row[xx] > 0) cnt++;
        }
    }
    return cnt;
}

//-------------------------------------------
// skeleton???? (cur, prev)?? ???????? ???? ??? ???
//-------------------------------------------
static bool findNextSkeletonPixel(const Mat& skel,
    const Point& cur,
    const Point& prev,
    Point& next)
{
    for (int dy = -1; dy <= 1; ++dy)
    {
        int yy = cur.y + dy;
        if (yy < 0 || yy >= skel.rows) continue;
        const uchar* row = skel.ptr<uchar>(yy);
        for (int dx = -1; dx <= 1; ++dx)
        {
            int xx = cur.x + dx;
            if (dx == 0 && dy == 0) continue;
            if (xx < 0 || xx >= skel.cols) continue;
            if (row[xx] == 0) continue;
            if (xx == prev.x && yy == prev.y) continue;
            next = Point(xx, yy);
            return true;
        }
    }
    return false;
}
// ???????? ??????? ��?? ?????? ???? ???
static void pruneShortBranches(cv::Mat& skel, int minBranchLen)
{
    cv::Mat visited = cv::Mat::zeros(skel.size(), CV_8UC1);
    bool removedSomething;

    do {
        removedSomething = false;
        visited.setTo(0);

        for (int y = 1; y < skel.rows - 1; ++y)
        {
            for (int x = 1; x < skel.cols - 1; ++x)
            {
                if (skel.at<uchar>(y, x) == 0) continue;
                if (visited.at<uchar>(y, x) != 0) continue;

                int deg = countWhiteNeighbors(skel, x, y);
                if (deg != 1) continue;   // ?????? ???? ???? ???? ???

                std::vector<cv::Point> branch;
                cv::Point start(x, y);
                cv::Point cur = start;
                cv::Point prev = start;

                while (true)
                {
                    branch.push_back(cur);
                    visited.at<uchar>(cur) = 255;

                    int d = countWhiteNeighbors(skel, cur.x, cur.y);

                    // ???????? ???? degree != 2 ??
                    //   -> ??? ???????? ?��???/?????? ????
                    if (!(cur == start) && d != 2)
                        break;

                    cv::Point next;
                    if (!findNextSkeletonPixel(skel, cur, prev, next))
                        break;   // ?? ?? ?? ?????? ????

                    prev = cur;
                    cur = next;
                }

                // ???? ???? ?��???/????????? (deg >= 3) ???? ???? ????
                if (!branch.empty())
                {
                    cv::Point last = branch.back();
                    int dlast = countWhiteNeighbors(skel, last.x, last.y);
                    if (dlast >= 3) {
                        branch.pop_back(); // junction?? ????��?
                    }
                }

                // ???? ???? ????? ??? ��???? ???? ??? ????
                if (!branch.empty() &&
                    (int)branch.size() < minBranchLen)
                {
                    for (auto& p : branch)
                        skel.at<uchar>(p) = 0;

                    removedSomething = true;
                }
            }
        }

    } while (removedSomething);
}

//-------------------------------------------
// distance transform ??? sub-pixel ??? ????
//-------------------------------------------
static cv::Point2f refineCenterSubpixel(const cv::Mat& dist, const cv::Point& p)
{
    CV_Assert(dist.type() == CV_32FC1);

    int x = p.x;
    int y = p.y;

    if (x <= 0 || x >= dist.cols - 1 ||
        y <= 0 || y >= dist.rows - 1)
    {
        return cv::Point2f((float)x, (float)y);
    }

    float f0 = dist.at<float>(y, x);
    float fx1 = dist.at<float>(y, x - 1);
    float fx2 = dist.at<float>(y, x + 1);
    float fy1 = dist.at<float>(y - 1, x);
    float fy2 = dist.at<float>(y + 1, x);

    auto quadVertex = [](float fm1, float f0, float fp1) -> float {
        float denom = (fm1 - 2.0f * f0 + fp1);
        if (std::fabs(denom) < 1e-6f) return 0.0f;
        float t = 0.5f * (fm1 - fp1) / denom;
        if (t > 1.0f)  t = 1.0f;
        if (t < -1.0f) t = -1.0f;
        return t;
        };

    float dx = quadVertex(fx1, f0, fx2);
    float dy = quadVertex(fy1, f0, fy2);

    return cv::Point2f(x + dx, y + dy);
}

//-------------------------------------------
// degree == 2 ?? ??????? ????? ???? ????(L??? ?????)
//-------------------------------------------
static bool getCornerAngleDeg(const Mat& skel, int x, int y, float& outDeg)
{
    if (skel.at<uchar>(y, x) == 0) return false;

    // ??? ?? ?? ??? 2?? ???
    Point n1(-1, -1), n2(-1, -1);
    int count = 0;

    for (int dy = -1; dy <= 1; ++dy)
    {
        int yy = y + dy;
        if (yy < 0 || yy >= skel.rows) continue;
        const uchar* row = skel.ptr<uchar>(yy);
        for (int dx = -1; dx <= 1; ++dx)
        {
            int xx = x + dx;
            if (dx == 0 && dy == 0) continue;
            if (xx < 0 || xx >= skel.cols) continue;
            if (row[xx] == 0) continue;

            if (count == 0) n1 = Point(xx, yy);
            else if (count == 1) { n2 = Point(xx, yy); }
            count++;
        }
    }

    if (count != 2) return false;

    Point2f v1 = Point2f((float)(n1.x - x), (float)(n1.y - y));
    Point2f v2 = Point2f((float)(n2.x - x), (float)(n2.y - y));

    float n1len = cv::norm(v1);
    float n2len = cv::norm(v2);
    if (n1len < 1e-3f || n2len < 1e-3f) return false;

    float cosTheta = (v1.x * v2.x + v1.y * v2.y) / (n1len * n2len);
    cosTheta = std::max(-1.0f, std::min(1.0f, cosTheta));

    float theta = std::acos(cosTheta);   // rad
    outDeg = theta * 180.0f / (float)CV_PI;

    return true;
}

//-------------------------------------------
// X/T/L ????/????? ??? (??????/???? ????)
//-------------------------------------------
cv::Point2f findIntersectionPointXTL(const cv::Mat& srcGray,
    int areaThresh = 50,
    int minBranchLen = 20,
    int searchRadius = 10,
    float cornerAngleThresh = 150.0f)
{
    CV_Assert(srcGray.type() == CV_8UC1);

    // 1) ????? (?? = 255, ??? = 0)
    Mat blur, bin;
    GaussianBlur(srcGray, blur, Size(5, 5), 1.0);
    threshold(blur, bin, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    // ?????????? ?????? ????
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(bin, bin, MORPH_OPEN, kernel);
    morphologyEx(bin, bin, MORPH_CLOSE, kernel);

    // ???? blob ????
    Mat labels, stats, centroids;
    int nLabels = connectedComponentsWithStats(bin, labels, stats, centroids);
    for (int i = 1; i < nLabels; ++i)
    {
        int area = stats.at<int>(i, CC_STAT_AREA);
        if (area < areaThresh)
            bin.setTo(0, labels == i);
    }

    // 2) ?????
    Mat skel;
    thinningZhangSuen(bin, skel);

    // 3) ???? pruning
    pruneShortBranches(skel, minBranchLen);


    // ???? ?????? ?��??? ???
    cv::Mat color;
    cv::cvtColor(srcGray, color, cv::COLOR_GRAY2BGR);

    // ??????? ????? ?????????? ????
    for (int y = 0; y < skel.rows; ++y)
    {
        const uchar* srow = skel.ptr<uchar>(y);
        cv::Vec3b* crow = color.ptr<cv::Vec3b>(y);

        for (int x = 0; x < skel.cols; ++x)
        {
            if (srow[x] > 0) {
                // BGR = (0,0,255) ?? ??????
                crow[x] = cv::Vec3b(0, 0, 255);
            }
        }
    }

    // ??��? ??????
    cv::imshow("skeleton on image", color);
    cv::waitKey(0);


    // 4) X/T junction + L corner ??? ????? ????
    Mat candMask = Mat::zeros(skel.size(), CV_8UC1);

    for (int y = 1; y < skel.rows - 1; ++y)
    {
        for (int x = 1; x < skel.cols - 1; ++x)
        {
            if (skel.at<uchar>(y, x) == 0) continue;

            int deg = countWhiteNeighbors(skel, x, y);

            // --- X / T ??? (deg >= 3) ---
            if (deg >= 3)
            {
                // minBranchLenXT : ?????? ?????? ??? ???? ????
                // angleMinXT/angleMaxXT : ?? ?? ???? ???? ????
                float minBranchLenXT = 10.0f;   // ???? ?��? ????
                float angleMinXT = 60.0f;
                float angleMaxXT = 120.0f;

                if (isXorTIntersectionCandidate(skel, x, y,
                    minBranchLenXT,
                    angleMinXT,
                    angleMaxXT))
                {
                    candMask.at<uchar>(y, x) = 255;
                }
            }
            // --- L ??? ??? (deg == 2) ---
            else if (deg == 2)
            {
                float angleDeg;
                if (getCornerAngleDegGlobal(skel, x, y, angleDeg, /*maxStep=*/10))
                {
                    // L?? 60~120??, ????(~180??)?? ????
                    if (angleDeg > 60.0f && angleDeg < 120.0f)
                    {
                        candMask.at<uchar>(y, x) = 255;
                    }
                }
            }
        }
    }

	cv::imshow("candidates", candMask);
    cv::waitKey(0);

    if (countNonZero(candMask) == 0)
    {
        // ????/??? ?? ??? ???
        return Point2f(0, 0);
    }

    // 4-1) ???? ? candidate ??????? ???? (????/???? 1????? ????)
    Mat cLabels, cStats, cCentroids;
    int cNum = connectedComponentsWithStats(candMask, cLabels, cStats, cCentroids);

    int bestLabel = -1;
    int bestArea = 0;
    for (int i = 1; i < cNum; ++i)
    {
        int area = cStats.at<int>(i, CC_STAT_AREA);
        if (area > bestArea)
        {
            bestArea = area;
            bestLabel = i;
        }
    }

    Point2f approx(0, 0);
    if (bestLabel <= 0)
    {
        vector<Point> pts;
        findNonZero(candMask, pts);
        for (auto& p : pts) {
            approx.x += p.x;
            approx.y += p.y;
        }
        approx.x /= (float)pts.size();
        approx.y /= (float)pts.size();
    }
    else
    {
        approx.x = (float)cCentroids.at<double>(bestLabel, 0);
        approx.y = (float)cCentroids.at<double>(bestLabel, 1);
    }

    // 5) distance transform + sub-pixel ??? ????
    Mat dist;
    distanceTransform(bin, dist, DIST_L2, 3);   // CV_32F

    Point bestPt(cvRound(approx.x), cvRound(approx.y));
    bestPt.x = std::clamp(bestPt.x, 0, dist.cols - 1);
    bestPt.y = std::clamp(bestPt.y, 0, dist.rows - 1);

    float bestVal = dist.at<float>(bestPt);

    for (int dy = -searchRadius; dy <= searchRadius; ++dy)
    {
        int yy = bestPt.y + dy;
        if (yy < 0 || yy >= dist.rows) continue;
        float* drow = dist.ptr<float>(yy);
        for (int dx = -searchRadius; dx <= searchRadius; ++dx)
        {
            int xx = bestPt.x + dx;
            if (xx < 0 || xx >= dist.cols) continue;
            float v = drow[xx];
            if (v > bestVal)
            {
                bestVal = v;
                bestPt.x = xx;
                bestPt.y = yy;
            }
        }
    }

    Point2f refined = refineCenterSubpixel(dist, bestPt);
    return refined;
}

int main() {
    cv::Mat src = cv::imread("D:\\Devs\\ClockImageReader\\clockimages\\cut-+.jpg", cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cout << "??????? ????? ?? ???????.\n";
        return -1;
    }
    cv::Mat grayOut, bin;

    if (src.channels() == 3)
        cv::cvtColor(src, grayOut, cv::COLOR_BGR2GRAY);
    else
        grayOut = src.clone();
    // ??? / ?? ???? ???? ?????? ???? ???
    Point2f cross = findIntersectionPointXTL(src,
        /*areaThresh=*/50,
        /*minBranchLen=*/60,
        /*searchRadius=*/10,
        /*cornerAngleThresh=*/150.0f);

    cout << "Intersection / Corner: " << cross << endl;

    Mat color;
    cvtColor(src, color, COLOR_GRAY2BGR);
    circle(color, cross, 5, Scalar(0, 0, 255), -1);
    imshow("result", color);
    waitKey(0);
    return 0;
}

```