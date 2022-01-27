// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(float x, float y, const Vector3f* _v)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    Vector3f p(x, y, 0);
    Vector3f a(_v[0].x(), _v[0].y(), 0);
    Vector3f b(_v[1].x(), _v[1].y(), 0);
    Vector3f c(_v[2].x(), _v[2].y(), 0);
    Vector3f ab = b - a;
    Vector3f bc = c - b;
    Vector3f ca = a - c;

    Vector3f ap = p - a;
    Vector3f bp = p - b;
    Vector3f cp = p - c;

    float z_ab_cross_ap = ab.cross(ap).z();
    float z_bc_cross_bp = bc.cross(bp).z();
    float z_ca_cross_cp = ca.cross(cp).z();
    if (z_ab_cross_ap * z_bc_cross_bp < 0 || z_ab_cross_ap * z_ca_cross_cp < 0 || z_bc_cross_bp * z_ca_cross_cp < 0) {
        return false;
    } else {
        return true;
    }

}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;  // 视口变换，将-1 -> n, 1 -> f，即为此式，参考：https://blog.csdn.net/wangdingqiaoit/article/details/51589825
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
    if (sampleMode == SampleMode::SSAA) {
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
                Vector3f color{0, 0, 0};
                for (int sampleX = 0; sampleX < superSampleCount; sampleX++) {
                    for (int sampleY = 0; sampleY < superSampleCount; sampleY++) {
                        int superSampleX = x * superSampleCount + sampleX;
                        int superSampleY = y * superSampleCount + sampleY;
                        int pixelIndex = get_super_sample_index(superSampleX, superSampleY);
                        color += super_sample_frame_buf[pixelIndex];
                    }
                }
                color /= superSampleCount * superSampleCount;
                set_pixel(Vector3f(x, y, 0), color);
            }
        }
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();
    float min_x = std::min(v[0].x(), std::min(v[1].x(), v[2].x()));
    float max_x = std::max(v[0].x(), std::max(v[1].x(), v[2].x()));
    float min_y = std::min(v[0].y(), std::min(v[1].y(), v[2].y()));
    float max_y = std::max(v[0].y(), std::max(v[1].y(), v[2].y()));
    min_x = std::floor(min_x);
    max_x = std::ceil(max_x);
    min_y = std::floor(min_y);
    max_y = std::ceil(max_y);

    if (sampleMode == SampleMode::Normal) {
        // TODO : Find out the bounding box of current triangle.
        // iterate through the pixel and find if the current pixel is inside the triangle
        for (int y = min_y; y <= max_y; y++) {
            for(int x = min_x; x < max_x; x++) {
                Vector3f point(x + 0.5, y + 0.5, 0);
                if (insideTriangle(point.x(), point.y(), t.v)) {
                    // If so, use the following code to get the interpolated z value.
                    auto[alpha, beta, gamma] = computeBarycentric2D(point.x(), point.y(), t.v);
                    float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;
                    int pixelIndex = get_index(point.x(), point.y());
                    if (depth_buf[pixelIndex] > z_interpolated) {
                        depth_buf[pixelIndex] = z_interpolated;
                        point.z() = z_interpolated;
                        set_pixel(point, t.getColor());
                    }

                    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
                }
            }
        }
    } else if (sampleMode == SampleMode::MSAA){
        std::vector<float> offset;
        for(int i = 0; i < superSampleCount; i++) {
            offset.push_back((i + 0.5f) / superSampleCount);
        }
        // TODO : Find out the bounding box of current triangle.
        // iterate through the pixel and find if the current pixel is inside the triangle
        for (int y = min_y; y <= max_y; y++) {
            for(int x = min_x; x < max_x; x++) {
                int count = 0;
                float minDepth = std::numeric_limits<float>::infinity();
                for (int sampleX = 0; sampleX < superSampleCount; sampleX++) {
                    for (int sampleY = 0; sampleY < superSampleCount; sampleY++) {
                        Vector3f point(x + offset[sampleX], y + offset[sampleY], 0);
                        if (insideTriangle(point.x(), point.y(), t.v)) {
                            count++;
                            // If so, use the following code to get the interpolated z value.
                            auto[alpha, beta, gamma] = computeBarycentric2D(point.x(), point.y(), t.v);
                            float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                            float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                            z_interpolated *= w_reciprocal;
                            if (minDepth > z_interpolated) {
                                minDepth = z_interpolated;
                            }
                        }
                    }
                }

                Vector3f color = t.getColor() * count / (superSampleCount * superSampleCount);
//                if (count > 0 && count < 4) {
//                    std::cout << "count: " << count << ", color: " << color << std::endl;
//                }
                int pixelIndex = get_index(x, y);
                if (depth_buf[pixelIndex] > minDepth) {
                    depth_buf[pixelIndex] = minDepth;
                    Vector3f point(x, y, minDepth);
                    set_pixel(point, color);
                }
            }
        }
    } else if (sampleMode == SampleMode::SSAA){
        std::vector<float> offset;
        for(int i = 0; i < superSampleCount; i++) {
            offset.push_back((i + 0.5f) / superSampleCount);
        }
        for (int y = min_y; y <= max_y; y++) {
            for(int x = min_x; x < max_x; x++) {
                for (int sampleX = 0; sampleX < superSampleCount; sampleX++) {
                    for (int sampleY = 0; sampleY < superSampleCount; sampleY++) {
                        Vector3f point(x + offset[sampleX], y + offset[sampleY], 0);
                        if (insideTriangle(point.x(), point.y(), t.v)) {
                            // If so, use the following code to get the interpolated z value.
                            auto[alpha, beta, gamma] = computeBarycentric2D(point.x(), point.y(), t.v);
                            float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                            float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                            z_interpolated *= w_reciprocal;
                            int superSampleX = x * superSampleCount + sampleX;
                            int superSampleY = y * superSampleCount + sampleY;
                            int pixelIndex = get_super_sample_index(superSampleX, superSampleY);
                            if (super_sample_depth_buf[pixelIndex] > z_interpolated) {
                                super_sample_depth_buf[pixelIndex] = z_interpolated;
                                set_super_sample_pixel(Vector3f(superSampleX, superSampleY, z_interpolated), t.getColor());
                            }
                        }
                    }
                }
            }
        }
    }

}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
        std::fill(super_sample_frame_buf.begin(), super_sample_frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
        std::fill(super_sample_depth_buf.begin(), super_sample_depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
    super_sample_frame_buf.resize(w * h * superSampleCount * superSampleCount);
    super_sample_depth_buf.resize(w * h * superSampleCount * superSampleCount);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

int rst::rasterizer::get_super_sample_index(int x, int y)
{
    return (2 * height - 1 - y) * 2 * width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

void rst::rasterizer::set_super_sample_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = get_super_sample_index(point.x(), point.y());
    super_sample_frame_buf[ind] = color;
}

// clang-format on