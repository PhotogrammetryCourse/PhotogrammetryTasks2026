#pragma once

#include <phg/sfm/defines.h>

#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>

struct vertex_info_t {
    std::vector<unsigned int> camera_ids;
    cv::Vec3b color;
    vector3d point_sum;
    double radius_sum;
    cv::Vec3d color_sum;
    size_t samples_count;
    bool is_bounding_box;
    size_t vertex_on_surface_id;

    vertex_info_t()
        : color(0, 0, 255) // red color, BGR convention (OpenCV compatible)
        , point_sum(0.0, 0.0, 0.0)
        , radius_sum(0.0)
        , color_sum(0.0, 0.0, 255.0)
        , samples_count(0)
        , is_bounding_box(false)
    {
    }

    vertex_info_t(unsigned int camera_id, const vector3d& point, double radius, const cv::Vec3b& color);

    void merge(const vertex_info_t& that);
    vector3d averagePoint(const vector3d& fallback) const;
    double averageRadius() const;
};

struct cell_info_t {
    size_t cell_id;

    float s_capacity;
    float t_capacity;
    float facets_capacities[4];

    cell_info_t()
        : cell_id(0)
        , s_capacity(0.0f)
        , t_capacity(0.0f)
        , facets_capacities { 0.0f, 0.0f, 0.0f, 0.0f }
    {
    }
};

typedef CGAL::Exact_predicates_inexact_constructions_kernel cgal_kernel_t;

typedef CGAL::Triangulation_vertex_base_with_info_3<vertex_info_t, cgal_kernel_t> cgal_vertex_t;
typedef CGAL::Triangulation_cell_base_with_info_3<cell_info_t, cgal_kernel_t> cgal_cell_t;

typedef CGAL::Triangulation_data_structure_3<cgal_vertex_t, cgal_cell_t, CGAL::Sequential_tag> triangulation_data_t;
typedef CGAL::Delaunay_triangulation_3<cgal_kernel_t, triangulation_data_t, CGAL::Fast_location> triangulation_t;

typedef cgal_kernel_t::Point_3 cgal_point_t;
typedef cgal_kernel_t::Triangle_3 cgal_triangle_t;

typedef triangulation_t::Vertex_handle vertex_handle_t;
typedef triangulation_t::Cell_handle cell_handle_t;

typedef triangulation_t::Facet cgal_facet_t;

vector3d from_cgal_point(cgal_point_t p);

cgal_point_t to_cgal_point(vector3d p);
