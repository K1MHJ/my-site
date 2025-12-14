// ConsoleApplication1.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
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
#include <queue>
#include <set>
#include <cmath>
#include <unordered_set>
#include "fb.h"

using namespace cv;
using namespace std;

namespace FA {


    static int countWhiteNeighbors(const Mat& img, int x, int y);
    static bool findNextSkeletonPixel(const Mat& skel,
        const Point& cur,
        const Point& prev,
        Point& next);

    struct BranchInfo
    {
        cv::Point2f dir; // 단위 방향 벡터
        float       length; // 픽셀 수
    };

    // (center -> firstNeighbor) 방향으로 가지를 따라가며 방향/길이를 구한다.
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
            if (deg != 2) // 끝점 or 분기점에서 멈춤
                break;
        }

        if (pts.size() < 2) return info;

        cv::Point2f v((float)(pts.back().x - center.x),
            (float)(pts.back().y - center.y));
        float n = std::sqrt(v.x * v.x + v.y * v.y);
        if (n < 1e-3f) return info;

        info.dir.x = v.x / n;
        info.dir.y = v.y / n;
        info.length = n; // center 기준 거리

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
    // X/T 후보 판정: 길이가 긴 가지 2개 이상 + 각도 충분히 큼
    static bool isXorTIntersectionCandidate(const cv::Mat& skel,
        int cx, int cy,
        float minBranchLenXT,
        float angleMinXT,
        float angleMaxXT)
    {
        if (skel.at<uchar>(cy, cx) == 0) return false;

        // 이웃 찾기
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

        // 각 이웃 방향으로 가지 추적
        std::vector<BranchInfo> branches;
        branches.reserve(neigh.size());
        for (auto& nb : neigh)
        {
            BranchInfo bi = traceBranch(skel, cv::Point(cx, cy), nb, /*maxStep=*/30);
            if (bi.length >= minBranchLenXT)
                branches.push_back(bi);
        }

        if (branches.size() < 2)
            return false; // 길이가 긴 가지가 2개 미만이면 노이즈로 간주

        // 가장 긴 두 가지 찾기
        std::sort(branches.begin(), branches.end(),
            [](const BranchInfo& a, const BranchInfo& b) {
                return a.length > b.length;
            });

        const cv::Point2f& d1 = branches[0].dir;
        const cv::Point2f& d2 = branches[1].dir;

        float ang = angleBetweenDeg(d1, d2);

        // X/T의 경우, 실제론 거의 직각 근처가 많을 테니 60~120도 정도를 허용
        if (ang >= angleMinXT && ang <= angleMaxXT)
            return true;

        return false;
    }

    // deg == 2 인 스켈레톤 픽셀의 "전역적인" 꺾임 각도 계산
    // 주변으로 여러 픽셀 따라가서 두 방향의 직선 방향을 추정
    static bool getCornerAngleDegGlobal(const cv::Mat& skel,
        int x, int y,
        float& outDeg,
        int maxStep = 10)
    {
        if (skel.at<uchar>(y, x) == 0) return false;

        // 1) 이웃 두 개 찾기
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
        if (neigh.size() != 2) return false; // deg != 2 이면 코너 아님

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
                    if (deg != 2) break; // 다음 분기점/끝점에서 멈춤
                }
            };

        // 2) 각 방향으로 점들을 모은 뒤, 첫/마지막 점으로 방향 벡터 추정
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
        dst /= 255;  // 0/1 로
        Mat prev = Mat::zeros(dst.size(), CV_8UC1);
        Mat diff;

        do {
            thinningIteration(dst, 0);
            thinningIteration(dst, 1);
            absdiff(dst, prev, diff);
            dst.copyTo(prev);
        } while (countNonZero(diff) > 0);

        dst *= 255;  // 다시 0/255
    }

    //-------------------------------------------
    // 8-이웃 흰 픽셀 개수
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
    // skeleton에서 (cur, prev)를 기준으로 다음 픽셀 찾기
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
    // 끝점에서 시작해서 짧은 가지를 잘라내는 함수
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
                    if (deg != 1) continue;   // 끝점이 아니면 가지 시작 아님

                    std::vector<cv::Point> branch;
                    cv::Point start(x, y);
                    cv::Point cur = start;
                    cv::Point prev = start;

                    while (true)
                    {
                        branch.push_back(cur);
                        visited.at<uchar>(cur) = 255;

                        int d = countWhiteNeighbors(skel, cur.x, cur.y);

                        // 시작점이 아니고 degree != 2 면
                        //   -> 다른 끝점이거나 분기점/교차점 도착
                        if (!(cur == start) && d != 2)
                            break;

                        cv::Point next;
                        if (!findNextSkeletonPixel(skel, cur, prev, next))
                            break;   // 더 갈 곳 없으면 종료

                        prev = cur;
                        cur = next;
                    }

                    // 가지 끝이 분기점/교차점이면 (deg >= 3) 삭제 대상에서 제외
                    if (!branch.empty())
                    {
                        cv::Point last = branch.back();
                        int dlast = countWhiteNeighbors(skel, last.x, last.y);
                        if (dlast >= 3) {
                            branch.pop_back(); // junction은 남겨두기
                        }
                    }

                    // 실제 가지 길이가 너무 짧으면 가지 픽셀 삭제
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
    // distance transform 기반 sub-pixel 중앙 보정
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
    // degree == 2 인 스켈레톤 픽셀의 꺾임 각도(L코너 검출용)
    //-------------------------------------------
    static bool getCornerAngleDeg(const Mat& skel, int x, int y, float& outDeg)
    {
        if (skel.at<uchar>(y, x) == 0) return false;

        // 이웃 중 흰 픽셀 2개 좌표
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
    // X/T/L 교차/코너점 찾기 (노이즈/굵기 고려)
    //-------------------------------------------
    cv::Point2f findIntersectionPointXTL(const cv::Mat& srcGray,
        int areaThresh = 50,
        int minBranchLen = 20,
        int searchRadius = 10,
        float cornerAngleThresh = 150.0f)
    {
        CV_Assert(srcGray.type() == CV_8UC1);

        // 1) 이진화 (선 = 255, 배경 = 0)
        Mat blur, bin;
        GaussianBlur(srcGray, blur, Size(5, 5), 1.0);
        threshold(blur, bin, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

        // 모폴로지로 노이즈 정리
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(bin, bin, MORPH_OPEN, kernel);
        morphologyEx(bin, bin, MORPH_CLOSE, kernel);

        // 작은 blob 제거
        Mat labels, stats, centroids;
        int nLabels = connectedComponentsWithStats(bin, labels, stats, centroids);
        for (int i = 1; i < nLabels; ++i)
        {
            int area = stats.at<int>(i, CC_STAT_AREA);
            if (area < areaThresh)
                bin.setTo(0, labels == i);
        }

        // 2) 세선화
        Mat skel;
        thinningZhangSuen(bin, skel);

        // 3) 가지 pruning
        pruneShortBranches(skel, minBranchLen);


        // 원본 그레이를 컬러로 변환
        cv::Mat color;
        cv::cvtColor(srcGray, color, cv::COLOR_GRAY2BGR);

        // 스켈레톤 픽셀을 빨간색으로 칠하기
        for (int y = 0; y < skel.rows; ++y)
        {
            const uchar* srow = skel.ptr<uchar>(y);
            cv::Vec3b* crow = color.ptr<cv::Vec3b>(y);

            for (int x = 0; x < skel.cols; ++x)
            {
                if (srow[x] > 0) {
                    // BGR = (0,0,255) → 빨간색
                    crow[x] = cv::Vec3b(0, 0, 255);
                }
            }
        }

        // 확인용 윈도우
        cv::imshow("skeleton on image", color);
        cv::waitKey(0);


        // 4) X/T junction + L corner 후보 마스크 생성
        Mat candMask = Mat::zeros(skel.size(), CV_8UC1);

        for (int y = 1; y < skel.rows - 1; ++y)
        {
            for (int x = 1; x < skel.cols - 1; ++x)
            {
                if (skel.at<uchar>(y, x) == 0) continue;

                int deg = countWhiteNeighbors(skel, x, y);

                // --- X / T 후보 (deg >= 3) ---
                if (deg >= 3)
                {
                    // minBranchLenXT : 교차로 인정할 최소 가지 길이
                    // angleMinXT/angleMaxXT : 두 선 사이 각도 범위
                    float minBranchLenXT = 10.0f;   // 해상도에 맞게 조정
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
                // --- L 코너 후보 (deg == 2) ---
                else if (deg == 2)
                {
                    float angleDeg;
                    if (getCornerAngleDegGlobal(skel, x, y, angleDeg, /*maxStep=*/10))
                    {
                        // L은 60~120도, 직선(~180도)은 제외
                        if (angleDeg > 60.0f && angleDeg < 120.0f)
                        {
                            candMask.at<uchar>(y, x) = 255;
                        }
                    }
                }
            }
        }

        if (countNonZero(candMask) == 0)
        {
            // 교차/코너 못 찾은 경우
            return Point2f(0, 0);
        }

        // 4-1) 가장 큰 candidate 컴포넌트 선택 (교차/코너가 1개라는 가정)
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

        // 5) distance transform + sub-pixel 중앙 보정
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


    static inline bool inb(const cv::Mat& m, int x, int y) {
        return (0 <= x && x < m.cols && 0 <= y && y < m.rows);
    }

    static std::vector<cv::Point> neighbors8(const cv::Mat1b& skel, const cv::Point& p) {
        static const int dx[8] = { -1,0,1,-1,1,-1,0,1 };
        static const int dy[8] = { -1,-1,-1,0,0,1,1,1 };
        std::vector<cv::Point> out;
        for (int k = 0; k < 8; ++k) {
            int nx = p.x + dx[k], ny = p.y + dy[k];
            if (inb(skel, nx, ny) && skel(ny, nx) != 0) out.emplace_back(nx, ny);
        }
        return out;
    }

    static int degree8(const cv::Mat1b& skel, const cv::Point& p) {
        return (int)neighbors8(skel, p).size();
    }
    //------------------------------
    // find nearest skeleton pixel around given point
    // optionally prefer junction (degree>=3)
    //------------------------------
    static cv::Point findNearestOnSkeleton(const cv::Mat1b& skel, cv::Point2f p, int r, bool preferJunction) {
        cv::Point best(-1, -1);
        float bestD2 = 1e30f;

        int cx = (int)std::lround(p.x);
        int cy = (int)std::lround(p.y);

        for (int dy = -r; dy <= r; ++dy) {
            for (int dx = -r; dx <= r; ++dx) {
                int x = cx + dx, y = cy + dy;
                if (!inb(skel, x, y)) continue;
                if (skel(y, x) == 0) continue;

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
                if (skel(y, x) == 0) continue;
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
    static std::vector<cv::Point> collectJunctionRegion(const cv::Mat1b& skel, const cv::Point& seed, int maxRadius) {
        std::vector<cv::Point> region;
        if (seed.x < 0) return region;

        cv::Mat1b vis = cv::Mat1b::zeros(skel.size());
        std::queue<cv::Point> q;

        auto push = [&](cv::Point pt) {
            if (!inb(skel, pt.x, pt.y)) return;
            if (vis(pt.y, pt.x)) return;
            if (skel(pt.y, pt.x) == 0) return;

            // radius constraint around seed
            int dx = pt.x - seed.x, dy = pt.y - seed.y;
            if (dx * dx + dy * dy > maxRadius * maxRadius) return;

            // keep mostly junction-ish pixels
            if (degree8(skel, pt) >= 3 || pt == seed) {
                vis(pt.y, pt.x) = 1;
                q.push(pt);
            }
            };

        push(seed);

        while (!q.empty()) {
            cv::Point p = q.front(); q.pop();
            region.push_back(p);
            for (auto nb : neighbors8(skel, p)) push(nb);
        }
        return region;
    }
    static int degree8Allowed(const cv::Mat1b& skel, const cv::Mat1b& allowed, const cv::Point& p)
    {
        int cnt = 0;
        for (auto nb : neighbors8(skel, p)) {
            if (allowed(nb.y, nb.x)) cnt++;
        }
        return cnt;
    }
    struct LongestPathResult {
        cv::Point endpoint;
        int length;                       // dist(endpoint)
        std::vector<cv::Point> path;      // optional (debug)
    };

    static LongestPathResult followLongestToEndpointBFS(
        const cv::Mat1b& skel,
        const cv::Mat1b& allowed,   // 0/1 mask : allowed==1인 곳만 사용
        const cv::Point& start)
    {
        const int W = skel.cols, H = skel.rows;

        cv::Mat1i dist(H, W, -1);
        cv::Mat2i parent(H, W, cv::Vec2i(-1, -1));

        std::queue<cv::Point> q;
        dist(start.y, start.x) = 0;
        q.push(start);

        cv::Point best = start;
        int bestD = 0;
        bool foundEndpoint = false;

        while (!q.empty()) {
            cv::Point cur = q.front(); q.pop();
            int d = dist(cur.y, cur.x);

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
                if (!allowed(nb.y, nb.x)) continue;
                if (dist(nb.y, nb.x) != -1) continue;
                dist(nb.y, nb.x) = d + 1;
                parent(nb.y, nb.x) = cv::Vec2i(cur.x, cur.y);
                q.push(nb);
            }
        }

        // path 복원(디버그/검증용)
        std::vector<cv::Point> path;
        {
            cv::Point cur = best;
            while (cur.x != -1) {
                path.push_back(cur);
                auto pr = parent(cur.y, cur.x);
                if (pr[0] == -1) break;
                cur = cv::Point(pr[0], pr[1]);
            }
            std::reverse(path.begin(), path.end());
        }

        return { best, bestD, path };
    }
    // regionSet을 마스크로 만든다 (region=0, 나머지 스켈레톤=1)
    static cv::Mat1b buildAllowedMask(const cv::Mat1b& skel, const std::vector<cv::Point>& region)
    {
        cv::Mat1b allowed = cv::Mat1b::zeros(skel.size());
        // 스켈레톤인 곳만 1
        for (int y = 0;y < skel.rows;y++)
            for (int x = 0;x < skel.cols;x++)
                if (skel(y, x)) allowed(y, x) = 1;

        // junction region 제거
        for (auto& p : region) allowed(p.y, p.x) = 0;
        return allowed;
    }

    // Find branch starts: neighbors of region pixels that are skeleton but NOT in region
    static std::vector<std::pair<cv::Point, cv::Point>> findBranchStarts(
        const cv::Mat1b& skel,
        const std::vector<cv::Point>& region)
    {
        std::unordered_set<int> regionSet;
        regionSet.reserve(region.size());
        for (auto& p : region) regionSet.insert(p.y * skel.cols + p.x);

        std::set<int> seenStarts;
        std::vector<std::pair<cv::Point, cv::Point>> starts; // (start, prevInsideRegion)

        for (auto& rp : region) {
            for (auto nb : neighbors8(skel, rp)) {
                int key = nb.y * skel.cols + nb.x;
                if (regionSet.count(key)) continue; // still in region
                if (seenStarts.insert(key).second) {
                    starts.push_back({ nb, rp });
                }
            }
        }
        return starts;
    }

    // Follow one branch until endpoint (degree==1) or no forward neighbor
    static cv::Point followToEndpoint(const cv::Mat1b& skel, cv::Point start, cv::Point prev, int maxSteps = 5000) {
        cv::Point cur = start;
        cv::Point prv = prev;

        for (int step = 0; step < maxSteps; ++step) {
            int deg = degree8(skel, cur);
            if (deg <= 1) return cur; // endpoint

            auto nbs = neighbors8(skel, cur);
            // remove prv
            nbs.erase(std::remove(nbs.begin(), nbs.end(), prv), nbs.end());
            if (nbs.empty()) return cur;

            // choose the neighbor that best continues direction (cur - prv)
            cv::Point2f dir = cv::Point2f((float)(cur.x - prv.x), (float)(cur.y - prv.y));
            float dirn = std::sqrt(dir.dot(dir)) + 1e-6f;

            int bestIdx = 0;
            float bestScore = -1e9f;

            for (int i = 0; i < (int)nbs.size(); ++i) {
                cv::Point2f v = cv::Point2f((float)(nbs[i].x - cur.x), (float)(nbs[i].y - cur.y));
                float vn = std::sqrt(v.dot(v)) + 1e-6f;
                float cosv = (v.dot(dir)) / (vn * dirn); // [-1,1]
                if (cosv > bestScore) { bestScore = cosv; bestIdx = i; }
            }

            cv::Point nxt = nbs[bestIdx];
            prv = cur;
            cur = nxt;
        }
        return cur;
    }

    // Main: given binary line mask + intersection point -> 4 endpoints
    static std::vector<cv::Point> findFourEndpointsFromIntersection(
        const cv::Mat& binLines, cv::Point2f intersection, int junctionSearchR = 10, int junctionRegionR = 6)
    {
        cv::Mat1b skel;
        thinningZhangSuen(binLines, skel);

        // 1) find a junction pixel near intersection (prefer degree>=3)
        cv::Point junc = findNearestOnSkeleton(skel, intersection, junctionSearchR, true);
        if (junc.x < 0) {
            // fallback: nearest skeleton pixel
            junc = findNearestOnSkeleton(skel, intersection, junctionSearchR, false);
        }

        // 2) collect junction region (to avoid "fat" junction)
        auto region = collectJunctionRegion(skel, junc, junctionRegionR);
        if (region.empty()) region.push_back(junc);

        // 3) branch starts = skeleton neighbors outside region
        auto starts = findBranchStarts(skel, region);

        auto allowed = buildAllowedMask(skel, region);

        // 4) follow each start to endpoint
        std::vector<cv::Point> endpoints;
        for (auto& sp : starts) {
            cv::Point start = sp.first;

            // start가 allowed가 아니면 스킵
            if (!allowed(start.y, start.x)) continue;

            auto res = followLongestToEndpointBFS(skel, allowed, start);
            endpoints.push_back(res.endpoint);
        }

        // (선이 2개 교차면 보통 4개가 나옴. 혹시 더 나오면 중복/근접 제거를 추가로 하면 됨.)
        return endpoints;
    }

    // Utility: label as left/right/up/down relative to junction (simple)
    static void classifyLRUD(const std::vector<cv::Point>& endpoints, cv::Point center,
        cv::Point& left, cv::Point& right, cv::Point& up, cv::Point& down)
    {
        left = right = up = down = cv::Point(-1, -1);

        int minX = INT_MAX, maxX = INT_MIN, minY = INT_MAX, maxY = INT_MIN;
        for (auto& p : endpoints) {
            if (p.x < minX) { minX = p.x; left = p; }
            if (p.x > maxX) { maxX = p.x; right = p; }
            if (p.y < minY) { minY = p.y; up = p; }
            if (p.y > maxY) { maxY = p.y; down = p; }
        }
    }
}
int main() {
    cv::Mat src = cv::imread("D:\\Devs\\ClockImageReader\\clockimages\\cut-+.jpg", cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cout << "이미지를 불러올 수 없습니다.\n";
        return -1;
    }
    cv::Mat grayOut, bin;

    if (src.channels() == 3)
        cv::cvtColor(src, grayOut, cv::COLOR_BGR2GRAY);
    else
        grayOut = src.clone();
    // 해상도 / 선 굵기에 따라 파라미터 조정 필요
    Point2f cross = FA::findIntersectionPointXTL(src,
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

    {
        int areaThresh = 50;
        // 1) 이진화 (선 = 255, 배경 = 0)
        Mat blur;
        GaussianBlur(src, blur, Size(5, 5), 1.0);
        threshold(blur, bin, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

        // 모폴로지로 노이즈 정리
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(bin, bin, MORPH_OPEN, kernel);
        morphologyEx(bin, bin, MORPH_CLOSE, kernel);

        // 작은 blob 제거
        Mat labels, stats, centroids;
        int nLabels = connectedComponentsWithStats(bin, labels, stats, centroids);
        for (int i = 1; i < nLabels; ++i)
        {
            int area = stats.at<int>(i, CC_STAT_AREA);
            if (area < areaThresh)
                bin.setTo(0, labels == i);
        }
        imshow("bin", bin);
        waitKey(0);
    }

    // 2) 교차점은 이미 구했다고 가정
    Point2f intersection(cross.x, cross.y);
    // 3) 끝점 4개 찾기
    //auto endpoints = FA::findFourEndpointsFromIntersection(bin, intersection);
    std::vector<cv::Point> endpoints;

    {
        ImageU8 bin_ = toImageU8(bin);
        FloatPoint intersection_(cross.x, cross.y);;
        int junctionSearchR = 10;
        int junctionRegionR = 6;


        ImageU8 skel;
        thinningZhangSuen(bin_, skel);

        // 1) find a junction pixel near intersection (prefer degree>=3)
        IntPoint junc = findNearestOnSkeleton(skel, intersection_, junctionSearchR, true);
        if (junc.x < 0) {
            // fallback: nearest skeleton pixel
            junc = findNearestOnSkeleton(skel, intersection_, junctionSearchR, false);
        }

        // 2) collect junction region (to avoid "fat" junction)
        auto region = collectJunctionRegion(skel, junc, junctionRegionR);
        if (region.empty()) region.push_back(junc);

        // 3) branch starts = skeleton neighbors outside region
        auto starts = findBranchStarts(skel, region);

        auto allowed = buildAllowedMask(skel, region);

        // 4) follow each start to endpoint
        std::vector<IntPoint> endpoints_;
        for (auto& sp : starts) {
            IntPoint start = sp.first;

            // start가 allowed가 아니면 스킵
            if (!allowed.at(start.x, start.y)) continue;

            auto res = followLongestToEndpointBFS(skel, allowed, start);
            endpoints_.push_back(res.endpoint);
        }
        for (auto p : endpoints_) {
            endpoints.push_back({ p.x, p.y });
        }
    }

    // 4) 좌/우/상/하 라벨(간단히)
    cv::Point left, right, up, down;
    FA::classifyLRUD(endpoints, cv::Point((int)intersection.x, (int)intersection.y), left, right, up, down);

    std::cout << "endpoints count=" << endpoints.size() << "\n";
    std::cout << "L=" << left << " R=" << right << " U=" << up << " D=" << down << "\n";

    // 5) 디버그 오버레이
    cv::Mat dbg;
    cv::cvtColor(src, dbg, cv::COLOR_GRAY2BGR);
    cv::circle(dbg, intersection, 4, { 0,0,255 }, -1);
    for (auto& p : endpoints) cv::circle(dbg, p, 4, { 0,255,0 }, -1);
    cv::imwrite("dbg_endpoints.png", dbg);

    return 0;
}
