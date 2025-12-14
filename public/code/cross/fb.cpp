#define _USE_MATH_DEFINES // for C++
#include <cmath>
#include "fb.h"
#include <unordered_set>

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
void thinningZhangSuen(const ImageU8& srcBin, ImageU8& dstBin)
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
FloatPoint findIntersectionPointXTL(const ImageU8& srcGray,
    int areaThresh,
    int minBranchLen,
    int searchRadius,
    float cornerAngleThresh/*현재는 사용 안 함*/)
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


static inline bool inb(const ImageU8& m, int x, int y) {
    return (0 <= x && x < m.width && 0 <= y && y < m.height);
}


static std::vector<IntPoint> neighbors8(const ImageU8& skel, const IntPoint& p) {
    static const int dx[8] = { -1,0,1,-1,1,-1,0,1 };
    static const int dy[8] = { -1,-1,-1,0,0,1,1,1 };
    std::vector<IntPoint> out;
    for (int k = 0; k < 8; ++k) {
        int nx = p.x + dx[k], ny = p.y + dy[k];
        if (inb(skel, nx, ny) && skel.at(nx, ny) != 0) out.emplace_back(nx, ny);
    }
    return out;
}

static int degree8(const ImageU8& skel, const IntPoint& p) {
    return (int)neighbors8(skel, p).size();
}

//------------------------------
    // find nearest skeleton pixel around given point
    // optionally prefer junction (degree>=3)
    //------------------------------
IntPoint findNearestOnSkeleton(ImageU8& skel, FloatPoint p, int r, bool preferJunction) 
{
    IntPoint best(-1, -1);
    float bestD2 = 1e30f;

    int cx = (int)std::lround(p.x);
    int cy = (int)std::lround(p.y);

    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            int x = cx + dx, y = cy + dy;
            if (!inb(skel, x, y)) continue;
            if (skel.at(x, y) == 0) continue;

            if (preferJunction && degree8(skel, { x,y }) < 3) continue;

            float ddx = x - p.x, ddy = y - p.y;
            float d2 = ddx * ddx + ddy * ddy;
            if (d2 < bestD2) {
                bestD2 = d2;
                best = { x, y };
            }
        }
    }

    if (best.x >= 0) return best;

    // fallback: any skeleton pixel (not necessarily junction)
    bestD2 = 1e30f;
    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            int x = cx + dx, y = cy + dy;
            if (!inb(skel, x, y)) continue;
            if (skel.at(x, y) == 0) continue;
            float ddx = x - p.x, ddy = y - p.y;
            float d2 = ddx * ddx + ddy * ddy;
            if (d2 < bestD2) { bestD2 = d2; best = { x,y }; }
        }
    }
    return best;
}

//------------------------------
// build junction region around seed:
// include pixels that are near seed and have degree>=3 (and their connected cluster)
//------------------------------
std::vector<IntPoint> collectJunctionRegion(const ImageU8& skel, const IntPoint& seed, int maxRadius) 
{
    std::vector<IntPoint> region;
    if (seed.x < 0) return region;

    ImageU8 vis(skel.width, skel.height);
    std::queue<IntPoint> q;

    auto push = [&](IntPoint pt) {
        if (!inb(skel, pt.x, pt.y)) return;
        if (vis.at(pt.x, pt.y)) return;
        if (skel.at(pt.x, pt.y) == 0) return;

        // radius constraint around seed
        int dx = pt.x - seed.x, dy = pt.y - seed.y;
        if (dx * dx + dy * dy > maxRadius * maxRadius) return;

        // keep mostly junction-ish pixels
        if (degree8(skel, pt) >= 3 || pt == seed) {
            vis.at(pt.x, pt.y) = 1;
            q.push(pt);
        }
        };

    push(seed);

    while (!q.empty()) {
        IntPoint p = q.front(); q.pop();
        region.push_back(p);
        for (auto nb : neighbors8(skel, p)) push(nb);
    }
    return region;
}
// Find branch starts: neighbors of region pixels that are skeleton but NOT in region
std::vector<std::pair<IntPoint, IntPoint>> findBranchStarts(
    const ImageU8& skel,
    const std::vector<IntPoint>& region)
{
    std::unordered_set<int> regionSet;
    regionSet.reserve(region.size());
    for (auto& p : region) regionSet.insert(p.y * skel.width + p.x);

    std::set<int> seenStarts;
    std::vector<std::pair<IntPoint, IntPoint>> starts; // (start, prevInsideRegion)

    for (auto& rp : region) {
        for (auto nb : neighbors8(skel, rp)) {
            int key = nb.y * skel.width + nb.x;
            if (regionSet.count(key)) continue; // still in region
            if (seenStarts.insert(key).second) {
                starts.push_back({ nb, rp });
            }
        }
    }
    return starts;
}
// regionSet을 마스크로 만든다 (region=0, 나머지 스켈레톤=1)
ImageU8 buildAllowedMask(const ImageU8& skel, const std::vector<IntPoint>& region)
{
    ImageU8 allowed(skel.width, skel.height);
    // 스켈레톤인 곳만 1
    for (int y = 0;y < skel.height;y++)
        for (int x = 0;x < skel.width;x++)
            if (skel.at(x, y)) allowed.at(x, y) = 1;

    // junction region 제거
    for (auto& p : region) allowed.at(p.x, p.y) = 0;
    return allowed;
}

static int degree8Allowed(const ImageU8& skel, const ImageU8& allowed, const IntPoint& p)
{
    int cnt = 0;
    for (auto nb : neighbors8(skel, p)) {
        if (allowed.at(nb.x, nb.y)) cnt++;
    }
    return cnt;
}

LongestPathResult followLongestToEndpointBFS(
    const ImageU8& skel,
    const ImageU8& allowed,   // 0/1 mask : allowed==1인 곳만 사용
    const IntPoint& start)
{
    const int W = skel.width, H = skel.height;

    ImageI32 dist(H, W, -1);
    std::vector<IntPoint> parent(static_cast<size_t>(W) * H, { -1,-1 });

    std::queue<IntPoint> q;
    dist.at(start.x, start.y) = 0;
    q.push(start);

    IntPoint best = start;
    int bestD = 0;
    bool foundEndpoint = false;

    while (!q.empty()) {
        IntPoint cur = q.front(); q.pop();
        int d = dist.at(cur.x, cur.y);

        // allowed 안에서 endpoint 판정(차수==1)
        int deg = degree8Allowed(skel, allowed, cur);
        if (cur != start && deg == 1) {
            // 가장 먼 endpoint 선택
            if (!foundEndpoint || d > bestD) {
                foundEndpoint = true;
                bestD = d;
                best = cur;
            }
        }
        else if (!foundEndpoint) {
            // endpoint를 못 찾는 특이 케이스(루프 등) 대비: 그냥 가장 먼 점
            if (d > bestD) {
                bestD = d;
                best = cur;
            }
        }

        for (auto nb : neighbors8(skel, cur)) {
            if (!allowed.at(nb.x, nb.y)) continue;
            if (dist.at(nb.x, nb.y) != -1) continue;
            dist.at(nb.x, nb.y) = d + 1;
            parent.at(nb.x + nb.y * W) = { cur.x, cur.y };
            q.push(nb);
        }
    }

    // path 복원(디버그/검증용)
    std::vector<IntPoint> path;
    {
        IntPoint cur = best;
        while (cur.x != -1) {
            path.push_back(cur);
            auto pr = parent.at(cur.y * W + cur.x);
            if (pr.x == -1) break;
            cur = IntPoint(pr.x, pr.y);
        }
        std::reverse(path.begin(), path.end());
    }

    return { best, bestD, path };
}