#include "min_cut_cgal_structs.h"
#include "min_cut_defines.h"

#include <libutils/rasserts.h>

cv::Vec3b averageColor(const cv::Vec3d& color_sum, size_t samples_count)
{
    rassert(samples_count > 0, 23781274121026);

    cv::Vec3b color;
    for (int i = 0; i < 3; ++i)
        color[i] = (unsigned char)(std::max(0, std::min((int)(std::lround(color_sum[i] / samples_count)), 255)));
    return color;
}

vector3d from_cgal_point(cgal_point_t p) { return vector3d(p.x(), p.y(), p.z()); }

cgal_point_t to_cgal_point(vector3d p) { return cgal_point_t(p[0], p[1], p[2]); }

vertex_info_t::vertex_info_t(unsigned int camera_id, const vector3d& point, double radius, const cv::Vec3b& color)
    : color(color)
    , point_sum(point)
    , radius_sum(radius)
    , color_sum(color[0], color[1], color[2])
    , samples_count(1)
    , is_bounding_box(false)
{
    camera_ids.push_back(camera_id);
}

void vertex_info_t::merge(const vertex_info_t& that)
{
    for (int i = 1; i < camera_ids.size(); ++i) {
        rassert(camera_ids[i - 1] < camera_ids[i], 23781274121024);
    }
    for (int i = 1; i < that.camera_ids.size(); ++i) {
        rassert(that.camera_ids[i - 1] < that.camera_ids[i], 23781274121021);
    }

    for (int i = 0; i < that.camera_ids.size(); ++i) {
        unsigned int ci = that.camera_ids[i];
        if (std::find(camera_ids.begin(), camera_ids.end(), ci) == camera_ids.end()) {
            camera_ids.push_back(ci);
        }
    }

    std::sort(camera_ids.begin(), camera_ids.end());
    for (int i = 1; i < camera_ids.size(); ++i) {
        rassert(camera_ids[i - 1] < camera_ids[i], 23781274121024);
    }

    if (that.samples_count > 0) {
        point_sum += that.point_sum;
        radius_sum += that.radius_sum;
        color_sum += that.color_sum;
        samples_count += that.samples_count;
#if MIN_CUT_AVERAGE_MERGED_VERTEX_COLORS
        color = averageColor(color_sum, samples_count);
#endif
    }
    is_bounding_box = is_bounding_box || that.is_bounding_box;
}

vector3d vertex_info_t::averagePoint(const vector3d& fallback) const
{
    if (samples_count == 0)
        return fallback;
    return point_sum * (1.0 / samples_count);
}

double vertex_info_t::averageRadius() const
{
    if (samples_count == 0)
        return 0.0;
    return radius_sum / samples_count;
}
