#ifndef ROTATE_RECT_OPS_H
#define ROTATE_RECT_OPS_H

#if defined(__CUDACC__) 
    #include <cuda_runtime.h>
    #define __DEVICE__ __device__
#else
    #define __DEVICE__
    #include <cmath>
    #include <algorithm>

    using std::isinf;
    using std::isnan;
    using std::max;
    using std::min;
#endif


// #define DEBUG


__DEVICE__ const int MAX_RECT_INTERSECTIONS = 8;  // MAX number of intersections between two rotated rectangles is 8

/**
 * Adapted from OpenCV imgproc/src/intersection.cpp
 * 
*/
enum RectIntersectTypes {
    INTERSECT_NONE = 0, //!< No intersection
    INTERSECT_PARTIAL  = 1, //!< partial intersection for both Rectangles (but no full intersection for either)
    INTERSECT_EQ  = 2, //!< Both Rectangles are identical
    INTERSECT_FULL_1  = 3, //!< Rectangle 1 is fully enclosed in Rectangle 2
    INTERSECT_FULL_2  = 4 //!< Rectangle 2 is fully enclosed in Rectangle 1
};

template <typename T>
__DEVICE__ inline T deg2rad(const T deg)
{
    return deg / 180.0 * 3.1415926535;
}

template <typename T>
__DEVICE__ inline bool check_rects_equal(const T* rect_pts1, const T* rect_pts2, const double samePointEps = 1e-5)
{
    for( int i = 0; i < 4; i++ )
    {
        if( fabs(rect_pts1[i*2] - rect_pts2[i*2]) > samePointEps || (fabs(rect_pts1[i*2+1] - rect_pts2[i*2+1]) > samePointEps) )
        {
            return false;
        }
    }
    return true;
}

template <typename T>
__DEVICE__ int compute_rect_line_intersects(const T* rect_pts1, const T* rect_pts2, 
        const T* vec1, const T* vec2, T* intersection)
{
    int num_intersects = 0;

    // Line test - test all line combos for intersection
    for( int i = 0; i < 4; i++ )
    {
        for( int j = 0; j < 4; j++ )
        {
            // Solve for 2x2 Ax=b
            T x21 = rect_pts2[j*2] - rect_pts1[i*2];
            T y21 = rect_pts2[j*2+1] - rect_pts1[i*2+1];

            T vx1 = vec1[i*2];
            T vy1 = vec1[i*2+1];

            T vx2 = vec2[j*2];
            T vy2 = vec2[j*2+1];

            T det = vx2*vy1 - vx1*vy2;

            T t1 = (vx2*y21 - vy2*x21) / det;
            T t2 = (vx1*y21 - vy1*x21) / det;

            // This takes care of parallel lines
            if( isinf(t1) || isinf(t2) || isnan(t1) || isnan(t2) )
            {
                continue;
            }

            if( t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f )
            {
                intersection[num_intersects*2] = rect_pts1[i*2] + vec1[i*2]*t1;
                intersection[num_intersects*2+1] = rect_pts1[i*2+1] + vec1[i*2+1]*t1;
                ++num_intersects;
            }
        }
    }
    return num_intersects;
}

template <typename T>
__DEVICE__ int compute_rect_vertices_intersects(const T* rect_pts1, const T* rect_pts2, const T* vec2, T* intersection)
{
    int num_intersects = 0;
    // Check for vertices from rect1 inside recct2
    for( int i = 0; i < 4; i++ )
    {
        // We do a sign test to see which side the point lies.
        // If the point all lie on the same sign for all 4 sides of the rect,
        // then there's an intersection
        int posSign = 0;
        int negSign = 0;

        T x = rect_pts1[i*2];
        T y = rect_pts1[i*2+1];

        for( int j = 0; j < 4; j++ )
        {
            // line equation: Ax + By + C = 0
            // see which side of the line this point is at
            T A = -vec2[j*2+1];
            T B = vec2[j*2];
            T C = -(A*rect_pts2[j*2] + B*rect_pts2[j*2+1]);

            T s = A*x + B*y + C;

            if( s >= 0 )
            {
                ++posSign;
            }
            else
            {
                ++negSign;
            }
        }

        if( posSign == 4 || negSign == 4 )
        {
            intersection[num_intersects*2] = rect_pts1[i*2];
            intersection[num_intersects*2+1] = rect_pts1[i*2+1];
            ++num_intersects;
        }
    }

    return num_intersects;
}

template <typename T>
__DEVICE__ int filter_duplicate_intersections(const int N, const T* in_intersection, T* out_intersection, const double samePointEps=1e-5, const int max_count=-1)
{
    if (max_count == 0)
        return 0;

    int count = 0;
    for( int i = 0; i < N; i++ )
    {
        int j;
        for( j = 0; j < count; j++ )
        {
            float dx = in_intersection[i*2] - out_intersection[j*2];
            float dy = in_intersection[i*2+1] - out_intersection[j*2+1];
            double d2 = dx*dx + dy*dy; // can be a really small number, need double here
            if( d2 < samePointEps*samePointEps ) // is duplicate
                break;
        }
        if (j == count)
        {
            out_intersection[count*2] = in_intersection[i*2];
            out_intersection[count*2+1] = in_intersection[i*2+1];
            count++;
            if (max_count != -1 && count == max_count)
                return count;
        }
    }
    return count;
}

/**
 * Adapted from OpenCV imgproc/src/intersection.cpp
 * 
*/
template <typename T>
__DEVICE__ inline float contourArea(const int npoints, const T* contour, bool oriented=false)
{
    if( npoints == 0 )
        return 0.;

    float a00 = 0.0;
    const T* prev = contour + (npoints-1)*2;
    for( int i = 0; i < npoints; i++ )
    {
        const T* p = contour + i*2;
        a00 += (float)prev[0] * p[1] - (float)prev[1] * p[0];
        prev = p;
    }

    a00 *= 0.5;
    if( !oriented )
        a00 = fabs(a00);

    return a00;
}



template <typename T>
__DEVICE__ inline void sort_hull_pts(const int n, T* pts)
{
    for (int i = 0; i < n; ++i) 
    {
        for (int j = 0; j < n; ++j)
        {
            T ax = pts[i*2];
            T ay = pts[i*2+1];
            T bx = pts[j*2];
            T by = pts[j*2+1];

            bool cond = ax<bx || (ax==bx && ay<by);
            if (cond)
            {
                pts[j*2] = ax;
                pts[j*2+1] = ay;
                pts[i*2] = bx;
                pts[i*2+1] = by;
            }
        }
    }
}

//Returns positive value if B lies to the left of OA, negative if B lies to the right of OA, 0 if collinear
template <typename T>
__DEVICE__ inline double cross(const T* O, const T* A, const T* B)
{
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0]);
}

template <typename T>
__DEVICE__ int convexHull(const int n, T* P, T* out)
{
    // adapted from https://www.hackerearth.com/practice/math/geometry/line-sweep-technique/tutorial/
    int k = 0;
    
    T H[2*MAX_RECT_INTERSECTIONS*2];
    
    // sort points
    sort_hull_pts(n, P);

    // Build lower hull
    for (int i = 0; i < n; ++i) {
        while (k >= 2 && cross(H + (k-2)*2, H + (k-1)*2, P + i*2) <= 0) k--;
        H[k*2] = P[i*2];
        H[k*2+1] = P[i*2+1];
        k++;
    }

    // Build upper hull
    //i starts from n-2 because n-1 is the point which both hulls will have in common
    //t=k+1 so that the upper hull has atleast two points to begin with
    for (int i = n-2, t = k+1; i >= 0; i--) {
        while (k >= t && cross(H + (k-2)*2, H + (k-1)*2, P + i*2) <= 0) k--;
        H[k*2] = P[i*2];
        H[k*2+1] = P[i*2+1];
        k++;
    }
    //the last point of upper hull is same with the fist point of the lower hull
    k = k - 1;
    for(size_t i = 0; i < k; i++)
    {
        out[i*2] = H[i*2];
        out[i*2+1] = H[i*2+1];
    }
    return k;
}

/**
 * Adapted from OpenCV imgproc/src/intersection.cpp
 * 
*/
template <typename T>
__DEVICE__ int rotatedRectangleIntersection( const T* rect_pts1, const T* rect_pts2, T* intersectingRegion, int& out_num_intersects )
{
    const float samePointEps = 0.00001f; // used to test if two points are the same

    // Specical case of rect1 == rect2
    {
        bool is_rects_eq = check_rects_equal(rect_pts1, rect_pts2, samePointEps);

        if(is_rects_eq)
        {
            for( int i = 0; i < 8; i++ )
            {
                intersectingRegion[i] = rect_pts1[i];
            }
            out_num_intersects = 4;
            return RectIntersectTypes::INTERSECT_EQ;
        }
    }

    out_num_intersects = 0;
    int ret = RectIntersectTypes::INTERSECT_NONE;

    // Line vector
    // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
    T vec1[8], vec2[8];
    for( int i = 0; i < 4; i++ )
    {
        vec1[i*2] = rect_pts1[((i+1)*2)%8] - rect_pts1[i*2];
        vec1[i*2+1] = rect_pts1[((i+1)*2)%8+1] - rect_pts1[i*2+1];

        vec2[i*2] = rect_pts2[((i+1)*2)%8] - rect_pts2[i*2];
        vec2[i*2+1] = rect_pts2[((i+1)*2)%8+1] - rect_pts2[i*2+1];
    }

    T intersectingRegion2[(4*4+4+4)*2];

    int num_intersects = 0;

    // Check for vertices from rect1 inside rect2
    num_intersects = compute_rect_vertices_intersects(rect_pts1, rect_pts2, vec2, intersectingRegion2 + out_num_intersects * 2);
    out_num_intersects += num_intersects;

#ifdef DEBUG
    printf("compute_rect_vertices_intersects 1 inside 2: %d\n", num_intersects);
#endif

    if (num_intersects == 4) // rect1 is fully inside rect2
    {
        return RectIntersectTypes::INTERSECT_FULL_1;
    }

    // Reverse the check - check for vertices from rect2 inside rect1
    num_intersects = compute_rect_vertices_intersects(rect_pts2, rect_pts1, vec1, intersectingRegion2 + out_num_intersects * 2);
    out_num_intersects += num_intersects;

#ifdef DEBUG
    printf("compute_rect_vertices_intersects 2 inside 1: %d\n", num_intersects);
#endif

    if (num_intersects == 4) // rect2 is fully inside rect1
    {
        return RectIntersectTypes::INTERSECT_FULL_2;
    }

    // Line test - test all line combos for intersection
    num_intersects = compute_rect_line_intersects(rect_pts1, rect_pts2, vec1, vec2, intersectingRegion2 + out_num_intersects * 2);
    out_num_intersects += num_intersects;

#ifdef DEBUG
    printf("compute_rect_line_intersects: %d\n", num_intersects);
#endif

    if( num_intersects != 0 )
    {
        ret = RectIntersectTypes::INTERSECT_PARTIAL;
    }

    num_intersects = filter_duplicate_intersections(out_num_intersects, intersectingRegion2, intersectingRegion, samePointEps, MAX_RECT_INTERSECTIONS);
    out_num_intersects = num_intersects;

#ifdef DEBUG
    printf("filter_duplicate_intersections: %d\n", num_intersects);
#endif

    if( num_intersects == 0 )
    {
        ret = RectIntersectTypes::INTERSECT_NONE;
    }

    return ret;
}


template <typename T>
__DEVICE__ float computeRectInterArea(const T* rect_pts1, const T* rect_pts2)
{
    T inter_pts_f[MAX_RECT_INTERSECTIONS*2];
    // int res = rotatedRectangleIntersection2(rect, pixel_rect, inter_pts);
    int num_intersects = 0;
    int res = rotatedRectangleIntersection(rect_pts1, rect_pts2, inter_pts_f, num_intersects);

#ifdef DEBUG
    printf("computeRectInterArea: num_intersect_pts: %d\n", num_intersects);
#endif

    float interArea = 0.0f;
    if (res == RectIntersectTypes::INTERSECT_NONE)
    {
        interArea = 0.0f;   
    } else if (res == RectIntersectTypes::INTERSECT_FULL_1 || res == RectIntersectTypes::INTERSECT_EQ)
    {
        interArea = contourArea(4, rect_pts1);
    } else if (res == RectIntersectTypes::INTERSECT_FULL_2)
    {
        interArea = contourArea(4, rect_pts2);
    } else {
        T order_pts_f2[MAX_RECT_INTERSECTIONS * 2];
        size_t npoints = convexHull(num_intersects, inter_pts_f, order_pts_f2);
        interArea = contourArea(npoints, order_pts_f2);
    }

    return interArea;
}


template <typename T>
__DEVICE__ inline void convert_region_to_pts(T const * const roi, T * pts)
{
    T cx = roi[0];
    T cy = roi[1];
    T w = roi[2];
    T h = roi[3];
    T angle = deg2rad(roi[4]);

    T b = cos(angle)*0.5f;
    T a = sin(angle)*0.5f;

    pts[0] = cx - a*h - b*w;
    pts[1] = cy + b*h - a*w;
    pts[2] = cx + a*h - b*w;
    pts[3] = cy - b*h - a*w;
    pts[4] = 2*cx - pts[0];
    pts[5] = 2*cy - pts[1];
    pts[6] = 2*cx - pts[2];
    pts[7] = 2*cy - pts[3];
}

template <typename T>
__DEVICE__ inline float computeRectIoU(T const * const region1, T const * const region2) 
{

  float area1 = region1[2] * region1[3];
  float area2 = region2[2] * region2[3];
  
  T rect_pts1[8*2];
  T rect_pts2[8*2];
  convert_region_to_pts(region1, rect_pts1);
  convert_region_to_pts(region2, rect_pts2);

  float area_inter = computeRectInterArea(rect_pts1, rect_pts2);

  float iou = area_inter / (area1 + area2 - area_inter + 1e-8);

#ifdef DEBUG
  printf("area1: %.3f, area2: %.3f, area_inter: %.3f, iou: %.3f\n",
      area1, area2, area_inter, iou);
#endif

  return iou;
}

template <typename T>
__DEVICE__ void compute_roi_pool_pts(const T* roi, T* out_pts, const float spatial_scale,
    const int pooled_height, const int pooled_width, const int pooled_height_idx, const int pooled_width_idx)
{
  int ph = pooled_height_idx;
  int pw = pooled_width_idx;

  // int roi_batch_ind = roi[0];
  T cx = roi[1] * spatial_scale;
  T cy = roi[2] * spatial_scale;
  T w = roi[3] * spatial_scale;
  T h = roi[4] * spatial_scale;
  T angle = deg2rad(roi[5]);

  // Force malformed ROIs to be 1x1
  w = max(w, 1.0f);
  h = max(h, 1.0f);

  //TransformPrepare
  T dx = -pooled_width/2.0;
  T dy = -pooled_height/2.0;
  T Sx = w / pooled_width;
  T Sy = h / pooled_height;
  T Alpha = cos(angle);
  T Beta = -sin(angle);
  T Dx = cx;
  T Dy = cy;

  T M[2][3]; 
  M[0][0] = Alpha*Sx;
  M[0][1] = Beta*Sy;
  M[0][2] = Alpha*Sx*dx+Beta*Sy*dy+Dx;
  M[1][0] = -Beta*Sx;
  M[1][1] = Alpha*Sy;
  M[1][2] = -Beta*Sx*dx+Alpha*Sy*dy+Dy;

  // ORDER IN CLOCKWISE OR ANTI-CLOCKWISE
  // (0,1),(0,0),(1,0),(1,1)
  out_pts[0] = M[0][0]*pw+M[0][1]*(ph+1)+M[0][2];
  out_pts[1] = M[1][0]*pw+M[1][1]*(ph+1)+M[1][2];
  out_pts[2] = M[0][0]*pw+M[0][1]*ph+M[0][2];
  out_pts[3] = M[1][0]*pw+M[1][1]*ph+M[1][2];
  out_pts[4] = M[0][0]*(pw+1)+M[0][1]*ph+M[0][2];
  out_pts[5] = M[1][0]*(pw+1)+M[1][1]*ph+M[1][2];
  out_pts[6] = M[0][0]*(pw+1)+M[0][1]*(ph+1)+M[0][2];
  out_pts[7] = M[1][0]*(pw+1)+M[1][1]*(ph+1)+M[1][2];

}

////////////////////////////////////////////////////////////////////////////////
// special cases for rotated box and axis-aligned box
////////////////////////////////////////////////////////////////////////////////

template <typename T>
__DEVICE__ inline bool is_same_rbox_aabox(
    const T rbox_pts[8],
    const T min_x,
    const T max_x,
    const T min_y,
    const T max_y,
    const double elapsed = 1e-5)
{
#pragma unroll
  for (int i = 0; i < 4; i++) {
    bool is_different_x = (fabs(rbox_pts[i*2] - min_x) > elapsed) && (fabs(rbox_pts[i*2] - max_x) > elapsed);
    bool is_different_y = (fabs(rbox_pts[i*2+1] - min_y) > elapsed) && (fabs(rbox_pts[i*2+1] - max_y) > elapsed);
    if (is_different_x || is_different_y) {
      return false;
    }
  }
  return true;
}

template <typename T>
__DEVICE__ inline bool is_duplicate_pt(
    const T lhs_x,
    const T lhs_y,
    const T rhs_x,
    const T rhs_y,
    const double elapsed = 1e-5
    )
{
  double d = (lhs_x - rhs_x) * (lhs_x - rhs_x) + (lhs_y - rhs_y) * (lhs_y - rhs_y);
  if (d < elapsed * elapsed) {
    return true;
  } else {
    return false;
  }
}

template <typename T>
__DEVICE__ inline bool is_point_in_rbox(
    const T rbox_pts[8],
    const T rbox_line_params[8],
    const T x,
    const T y
    )
{
  // We do a sign test to see which side the point lies.
  // If the point all lie on the same sign for all 4 sides of the rect,
  // then there's an intersection
  int num_pos_sign = 0;
  int num_neg_sign = 0;

  for(int j = 0; j < 4; j++) {
    // line equation: Ax + By + C = 0
    // see which side of the line this point is at
    T A = -rbox_line_params[j*2+1];
    T B = rbox_line_params[j*2];
    T C = -(A * rbox_pts[j*2] + B * rbox_pts[j*2+1]);
    T s = A*x + B*y + C;

    if (s >= 0) {
      ++ num_pos_sign;
    } else {
      ++ num_neg_sign;
    }
  }

  return (num_pos_sign == 4 || num_neg_sign == 4);
}

template <typename T>
__DEVICE__ inline bool is_point_in_aabox(
    const T rbox_x,
    const T rbox_y,
    const T min_x,
    const T max_x,
    const T min_y,
    const T max_y
    )
{
  if (rbox_x < min_x || rbox_x > max_x) {
    return false;
  }
  if (rbox_y < min_y || rbox_y > max_y) {
    return false;
  }
  return true;
}

template <typename T>
__DEVICE__ inline bool is_rbox_in_aabox(
    const T rbox_pts[8],
    const T min_x,
    const T max_x,
    const T min_y,
    const T max_y
    )
{
#pragma unroll
  for (int i = 0; i < 4; i++) {
    if (rbox_pts[i*2] < min_x || rbox_pts[i*2] > max_x) {
      return false;
    }
    if (rbox_pts[i*2+1] < min_y || rbox_pts[i*2+1] > max_y) {
      return false;
    }
  }
  return true;
}

template <typename T>
__DEVICE__ inline bool get_itersect_y_aabox(
    T& y,
    const T rbox_pts[8],
    const int rbox_pt_idx,
    const T x,
    const T min_y,
    const T max_y
    )
{
  T start_x = rbox_pts[rbox_pt_idx*2];
  T end_x = rbox_pts[(rbox_pt_idx+1)*2 % 8];
  T start_y = rbox_pts[rbox_pt_idx*2 + 1];
  T end_y = rbox_pts[(rbox_pt_idx+1)*2 % 8 + 1];

  if (start_x == end_x) {
    return false;
  }

  if ((x - start_x)*(x - end_x) > 0) {
    return false;
  }

  T slope = (end_y - start_y) / (end_x - start_x);
  y = start_y + slope * (x - start_x);

  if (y <= min_y || y >= max_y) {
    return false;
  }
  if ((y - start_y)*(y - end_y) > 0) {
    return false;
  }

  // NOTE: discard the line intersection point
  // if it's the same as one of the vertex of one rectangle
  if (is_duplicate_pt(x, y, x, min_y) || is_duplicate_pt(x, y, x, max_y)) {
    return false;
  }
  if (is_duplicate_pt(x, y, start_x, start_y) || is_duplicate_pt(x, y, end_x, end_y)) {
    return false;
  }

  return true;
}

template <typename T>
__DEVICE__ inline bool get_itersect_x_aabox(
    T& x,
    const T rbox_pts[8],
    const int rbox_pt_idx,
    const T y,
    const T min_x,
    const T max_x
    )
{
  T start_x = rbox_pts[rbox_pt_idx*2];
  T end_x = rbox_pts[(rbox_pt_idx+1)*2 % 8];
  T start_y = rbox_pts[rbox_pt_idx*2 + 1];
  T end_y = rbox_pts[(rbox_pt_idx+1)*2 % 8 + 1];

  if (start_y == end_y) {
    return false;
  }

  if ((y - start_y)*(y - end_y) > 0) {
    return false;
  }

  T rev_slope = (end_x - start_x) / (end_y - start_y);
  x = start_x + rev_slope * (y - start_y);

  if (x <= min_x || x >= max_x) {
    return false;
  }
  if ((x - start_x)*(x - end_x) > 0) {
    return false;
  }

  // NOTE: discard the line intersection point
  // if it's the same as one of the vertex of one rectangle
  if (is_duplicate_pt(x, y, min_x, y) || is_duplicate_pt(x, y, max_x, y)) {
    return false;
  }
  if (is_duplicate_pt(x, y, start_x, start_y) || is_duplicate_pt(x, y, end_x, end_y)) {
    return false;
  }

  return true;
}

template <typename T>
__DEVICE__ inline T polygon_area(const int num_vertices, const T* vertices)
{
  if (num_vertices == 0) {
    return T(0);
  }

  T area = T(0);

  const T* prev = vertices + (num_vertices - 1) * 2;
  for (int i = 0; i < num_vertices; i++) {
    const T* curr = vertices + i * 2;
    area += prev[0] * curr[1] - prev[1] * curr[0];
    prev = curr;
  }

  area = fabs(area * 0.5);

  return area;
}

// Refer to: https://stackoverflow.com/questions/6989100/sort-points-in-clockwise-order
template <typename T>
__DEVICE__ inline bool anti_clockwise_less_than(
    const T lhs_x,
    const T lhs_y,
    const T rhs_x,
    const T rhs_y,
    const T center_x,
    const T center_y
    )
{
  if (lhs_x - center_x >= 0 && rhs_x - center_x < 0) {
    return false;
  }

  if (lhs_x - center_x < 0 && rhs_x - center_x >= 0) {
    return true;
  }

  if (lhs_x - center_x == 0 && rhs_x - center_x == 0) {
    if (lhs_y - center_y >= 0 || rhs_y - center_y >= 0) {
      return lhs_y <= rhs_y;
    }
    return rhs_y <= lhs_y;
  }

  // compute the cross product of vectors: (center -> lhs) x (center -> rhs)
  T det = (lhs_x - center_x) * (rhs_y - center_y) - (rhs_x - center_x) * (lhs_y - center_y);
  if (det < 0) {
    return false;
  }
  if (det > 0) {
    return true;
  }

  // points lhs and rhs are on the same line from the center
  // check which point is closer to the center
  T lhs_d = (lhs_x - center_x) * (lhs_x - center_x) + (lhs_y - center_y) * (lhs_y - center_y);
  T rhs_d = (rhs_x - center_x) * (rhs_x - center_x) + (rhs_y - center_y) * (lhs_y - center_y);
  return lhs_d <= rhs_d;
}

template <typename T>
__DEVICE__ inline void get_center_point(
    T& center_x,
    T& center_y,
    const int num_pts,
    T* pts
    )
{
  center_x = T(0);
  center_y = T(0);
  for (int i = 0; i < num_pts; ++i) {
    center_x += pts[i*2];
    center_y += pts[i*2+1];
  }
  center_x /= num_pts;
  center_y /= num_pts;
}

template <typename T>
__DEVICE__ inline void anti_clockwise_sort(const int num_pts, T* pts)
{
  T center_x;
  T center_y;
  get_center_point(center_x, center_y, num_pts, pts);

  for (int i = 0; i < num_pts; ++i) {
    for (int j = 0; j < num_pts; ++j) {
      T lhs_x = pts[i*2];
      T lhs_y = pts[i*2+1];
      T rhs_x = pts[j*2];
      T rhs_y = pts[j*2+1];
      if (anti_clockwise_less_than(lhs_x, lhs_y, rhs_x, rhs_y, center_x, center_y)) {
        pts[i*2] = rhs_x;
        pts[i*2+1] = rhs_y;
        pts[j*2] = lhs_x;
        pts[j*2+1] = lhs_y;
      }
    }
  }
}

template <typename T>
__DEVICE__ T itersect_area_rbox_aabox(
    const T rbox_pts[8],
    const T rbox_area,
    const T min_x,
    const T max_x,
    const T min_y,
    const T max_y
    )
{
  T intersection_pts[MAX_RECT_INTERSECTIONS * 2];
  int num_intersect_pts = 0;

  T area = 0;

  // rbox is the same as aabox
  if (is_same_rbox_aabox(rbox_pts, min_x, max_x, min_y, max_y)) {
    area = (max_x - min_x) * (max_y - min_y);
    return area;
  }

  // line parameters for rbox
  // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
  T line_params[8];
  for (int i = 0; i < 4; i++) {
    line_params[i*2] = rbox_pts[((i+1)*2) % 8] - rbox_pts[i*2];
    line_params[i*2+1] = rbox_pts[((i+1)*2) % 8 + 1] - rbox_pts[i*2 + 1];
  }

  // check whether corner point of aabox is inside the rbox
  // Manually unroll
  int old_num_intersect_pts = num_intersect_pts;
  if (is_point_in_rbox(rbox_pts, line_params, min_x, min_y)) {
    intersection_pts[num_intersect_pts*2] = min_x;
    intersection_pts[num_intersect_pts*2+1] = min_y;
    ++ num_intersect_pts;
  }
  if (is_point_in_rbox(rbox_pts, line_params, min_x, max_y)) {
    intersection_pts[num_intersect_pts*2] = min_x;
    intersection_pts[num_intersect_pts*2+1] = max_y;
    ++ num_intersect_pts;
  }
  if (is_point_in_rbox(rbox_pts, line_params, max_x, min_y)) {
    intersection_pts[num_intersect_pts*2] = max_x;
    intersection_pts[num_intersect_pts*2+1] = min_y;
    ++ num_intersect_pts;
  }
  if (is_point_in_rbox(rbox_pts, line_params, max_x, max_y)) {
    intersection_pts[num_intersect_pts*2] = max_x;
    intersection_pts[num_intersect_pts*2+1] = max_y;
    ++ num_intersect_pts;
  }

  // aabox is totally inside rbox
  if (num_intersect_pts - old_num_intersect_pts == 4) {
    area = (max_x - min_x) * (max_y - min_y);
#ifdef DEBUG
    printf("aabox is totally inside rbox: %f\n", area);
#endif
    return area;
  }

  old_num_intersect_pts = num_intersect_pts;
  for (int i = 0; i < 4; i++) {
    if (is_point_in_aabox(rbox_pts[i*2], rbox_pts[i*2+1], min_x, max_x, min_y, max_y)) {
      intersection_pts[num_intersect_pts*2] = rbox_pts[i*2];
      intersection_pts[num_intersect_pts*2+1] = rbox_pts[i*2+1];
      ++ num_intersect_pts;
    }
  }

  // rbox is totally inside aabox
  if (num_intersect_pts - old_num_intersect_pts == 4) {
// if (is_rbox_in_aabox(rbox_pts, min_x, max_x, min_y, max_y)) {
    area = rbox_area;
#ifdef DEBUG
    printf("rbox is totally inside aabox: %f %f\n", area, rbox_area);
#endif
    return area;
  }

#ifdef DEBUG
  printf("Before line test: num_intersect_pts: %d\n", num_intersect_pts);
#endif

  // line test - test all line combos for intersection
  T x;
  T y;
  for (int i = 0; i < 4; i++) {
    if (get_itersect_y_aabox(y, rbox_pts, i, min_x, min_y, max_y)) {
      intersection_pts[num_intersect_pts*2] = min_x;
      intersection_pts[num_intersect_pts*2+1] = y;
      ++ num_intersect_pts;
    }

    if (get_itersect_y_aabox(y, rbox_pts, i, max_x, min_y, max_y)) {
      intersection_pts[num_intersect_pts*2] = max_x;
      intersection_pts[num_intersect_pts*2+1] = y;
      ++ num_intersect_pts;
    }

    if (get_itersect_x_aabox(x, rbox_pts, i, min_y, min_x, max_x)) {
      intersection_pts[num_intersect_pts*2] = x;
      intersection_pts[num_intersect_pts*2+1] = min_y;
      ++ num_intersect_pts;
    }

    if (get_itersect_x_aabox(x, rbox_pts, i, max_y, min_x, max_x)) {
      intersection_pts[num_intersect_pts*2] = x;
      intersection_pts[num_intersect_pts*2+1] = max_y;
      ++ num_intersect_pts;
    }
  }

#ifdef DEBUG
  printf("After line test: num_intersect_pts: %d\n", num_intersect_pts);
#endif

#if 0
  // NOTE: don't have to filter duplicate intersections again
  const float elapsed = 0.00001f;
  T out_intersection_pts[MAX_RECT_INTERSECTIONS * 2];
  num_intersect_pts = filter_duplicate_intersections(num_intersect_pts, intersection_pts, out_intersection_pts, elapsed, MAX_RECT_INTERSECTIONS);
#endif

  if (num_intersect_pts < 3) {
    area = T(0);
  } else {
    // sort the points of intersection
#if 1
    anti_clockwise_sort(num_intersect_pts, intersection_pts);
    area = polygon_area(num_intersect_pts, intersection_pts);
#else
    anti_clockwise_sort(num_intersect_pts, out_intersection_pts);
    area = polygon_area(num_intersect_pts, out_intersection_pts);
#endif
  }

  return area;
}

#endif /* ROTATE_RECT_OPS_H */
