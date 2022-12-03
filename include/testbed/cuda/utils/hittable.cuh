#pragma once
#ifndef TESTBED_CUDA_UTILS_HITTABLE_H_
#define TESTBED_CUDA_UTILS_HITTABLE_H_

/**
 * @file: include/testbed/cuda/hittable.cuh
 * @author: sailing-innocent
 * @create: 2022-12-03
 * @desp: The hittable class for tracer
*/

#include <testbed/cuda/utils/ray.cuh>

ING_NAMESPACE_BEGIN

struct hit_record {
    point p;
    vec4 normal;
    float t;
    bool front_face;

    ING_CU_HOST_DEVICE inline void set_face_normal(const Ray& r, const vec4& outward_normal) {
        front_face = dot(r.dir(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class Sphere {
public:
    ING_CU_HOST_DEVICE Sphere() {}
    ING_CU_HOST_DEVICE bool hit(
        const Ray& r, float t_min, float t_max, hit_record& rec
    ) const {
        vec4 oc = r.origin() - center;
        float a = dot(r.dir(), r.dir());
        float halfb = dot(oc, r.dir());
        float c = dot(oc, oc) - radius * radius;
        auto discriminant = halfb * halfb -  a * c;
        if (discriminant < 0) return false;
        float sqrtd = sqrtf(discriminant);

        auto root = (-halfb-sqrtd)/a;
        if (root < t_min || t_max < root) {
            root = (-halfb+sqrtd)/a;
            if ( root < t_min || t_max < root)
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec4 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);

        return true;
    };
public:
    point center{0.0f, 0.0f, -1.0f};
    float radius = 0.5f;
};

ING_NAMESPACE_END

#endif // TESTBED_CUDA_UTILS_HITTABLE_H_