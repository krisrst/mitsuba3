
#pragma once
#include <math.h>
#include <mitsuba/render/optix/common.h>
#include <mitsuba/render/optix/math.cuh>


#define NUM_POLY_TERMS 10

struct OptixPolyData {
    optix::BoundingBox3f bbox;
    optix::Transform4f to_world;
    optix::Transform4f to_object;
    optix::Vector3f center;

    float k;
    float c;
    float r;
    float h_lim;
    float poly_coefs[NUM_POLY_TERMS];
    bool  poly_is_even;
    float z_min;
    float z_max;
    float z_min_base;
    float z_max_base;

    bool flip_normals;
};

#ifdef __CUDACC__

bool __device__ point_within_surf_bounds( Vector3f point, Vector3f center, float z_min, float z_max, float h_lim)
{
    Vector3f p = point - center;

    float h = sqrt( sqr(p[0]) + sqr(p[1]));
    float z = p[2];

    return (h <= h_lim) && (z >= z_min) && (z <= z_max);
}

bool __device__ find_sphere_intersections( float &near_t, float &far_t,
                          Vector3f center,
                          float c, float k,
                          const Ray3f &ray)
{

    // Unit vector
    Vector3f d = ray.d;

    // Origin
    Vector3f o = ray.o;

    // Center of sphere
    Vector3f cs = center;

    float dx = d[0], dy = d[1], dz = d[2];
    float ox = o[0] - cs[0], oy = o[1] - cs[1], oz = o[2] - cs[2];

    float g = -1 * ( 1 + k );

    float A = -1 * g * sqr(dz) + sqr(dx) + sqr(dy);
    float B = -1 * g * 2 * oz * dz + 2 * ox * dx  + 2 * oy * dy - 2 * dz / c;
    float C = -1 * g * sqr(oz) + sqr(ox) + sqr(oy) - 2 * oz / c;

    bool solution_found = solve_quadratic(A, B, C, near_t, far_t);

    return solution_found;
}

bool __device__ find_conic_intersection( float &golden_t,
                          Vector3f center,
                          float c, float k,
                          float z_min_base, float z_max_base, float h_lim,
                          const Ray3f &ray)
{
    float near_t0, far_t0;
    bool solution = find_sphere_intersections( near_t0, far_t0,
                                               center,
                                               c, k,
                                               ray);

    if(!solution) {
        return false;
    }

    // Is one or both hits on the sphere surface which is limited by lens height & depth?
    bool valid_near = point_within_surf_bounds( ray(near_t0),
                                                center,
                                                z_min_base, z_max_base, h_lim );

    valid_near = valid_near && (near_t0 >= 0.f && near_t0 < ray.maxt);

    bool valid_far = point_within_surf_bounds( ray(far_t0),
                                               center,
                                               z_min_base, z_max_base, h_lim );

    valid_far = valid_far && (far_t0 >= 0.f && far_t0 < ray.maxt);

    if(!(valid_far || valid_near)) {
        return false;
    }

    if(valid_near) {
        golden_t = near_t0;
    }
    else {
        golden_t = far_t0;
    }
    return true;
}

float __device__ conic_sag(float r, float c, float kappa)
{
    return c * sqr(r) / (1 + sqrt(1 - (1+kappa) * sqr(c) * sqr(r)));
}

Vector3f __device__ conic_normal_vector(float x, float y, float c, float kappa)
{
    float dx = (x * c) / sqrt(1 - (1+kappa) * (sqr(x) * sqr(c)));
    float dy = (y * c) / sqrt(1 - (1+kappa) * (sqr(y) * sqr(c)));
    float dz = -1.0;

    return Vector3f(dx, dy, dz);
}

float __device__ aspheric_radial_polyterms(float r, float poly[NUM_POLY_TERMS])
{
    float dz = 0;
    float ri = r;
    for(size_t i=0; i < NUM_POLY_TERMS; i++) {
        dz += poly[i]*ri;
        ri *= r;
    }
    return dz;
}

float __device__ aspheric_even_polyterms(float r, float poly[NUM_POLY_TERMS])
{
    float dz = 0;
    float ri = r*r;
    for(size_t i=0; i < NUM_POLY_TERMS; i++) {
        dz += poly[i]*ri;
        ri *= r*r;
    }
    return dz;
}

float __device__ aspheric_polyterms(float r, float poly[NUM_POLY_TERMS], bool poly_is_even)
{
    if(poly_is_even) {
        return aspheric_even_polyterms(r, poly);
    }
    else {
        return aspheric_radial_polyterms(r, poly);
    }
}

Vector3f __device__ aspheric_radial_polyterms_derivatives(float x, float y, float poly[NUM_POLY_TERMS])
{
    float r = sqrt( sqr(x) + sqr(y));

    float dr = 0;
    float ri = 1/r; // starting at 1/r because we scale with x, y later
    for(size_t i=0; i < NUM_POLY_TERMS; i++) {
        dr += (i+1)*poly[i]*ri;
        ri *= r;
    }

    return Vector3f(dr*x, dr*y, 0);
}

Vector3f __device__ aspheric_even_polyterms_derivatives(float x, float y, float poly[NUM_POLY_TERMS])
{
    float r = sqrt( sqr(x) + sqr(y));

    float dr = 0;
    float ri = 1; // starting at r/r because we scale with x, y later
    for(size_t i=0; i < NUM_POLY_TERMS; i++) {
        dr += 2*(i+1)*poly[i]*ri;
        ri *= r*r;
    }

    return Vector3f(dr*x, dr*y, 0);
}

Vector3f __device__ aspheric_polyterms_derivatives(float x, float y, float poly[NUM_POLY_TERMS], bool poly_is_even)
{
    if(poly_is_even) {
        return aspheric_even_polyterms_derivatives(x, y, poly);
    }
    else {
        return aspheric_radial_polyterms_derivatives(x, y, poly);
    }
}

float __device__ polyasphsurf_implicit_fun(Vector3f P, Vector3f center, float curvature, float kappa, float poly[NUM_POLY_TERMS], bool poly_is_even)
{
    float x = P[0] - center[0], y = P[1] - center[1], z = P[2] - center[2];
    float r = sqrt( sqr(x) + sqr(y));

    float sag = conic_sag(r, curvature, kappa) + aspheric_polyterms(r, poly, poly_is_even);
    return sag - z;
}

Vector3f __device__ polyasphsurf_normal_vector(Vector3f P, Vector3f center, float curvature, float kappa, float poly[NUM_POLY_TERMS], bool poly_is_even)
{
    float x = P[0] - center[0], y = P[1] - center[1], z = P[2] - center[2];

    return conic_normal_vector(x, y, curvature, kappa) + aspheric_polyterms_derivatives(x, y, poly, poly_is_even);
}


// Based on the "Spencer and Murty" general ray tracing procedure
extern "C" __global__ void __intersection__poly()
{
    const OptixHitGroupData *sbt_data = (OptixHitGroupData*) optixGetSbtDataPointer();
    OptixPolyData *asurf = (OptixPolyData *)sbt_data->data;

    // Ray in instance-space
    Ray3f ray = get_ray();

    float t;
    if(!find_conic_intersection(t, asurf->center, asurf->c, asurf->k, asurf->z_min_base, asurf->z_max_base, asurf->h_lim, ray)) {
        return;
    }

    Vector3f P = ray(t);
    float e = polyasphsurf_implicit_fun(P, asurf->center, asurf->c, asurf->k, asurf->poly_coefs, asurf->poly_is_even);

    float ae_min = abs(e);
    float t_min = t;

    float tolerance = 1e-6;
    unsigned int iter = 0;
    while( abs(e) > tolerance && iter < 8) {
        Vector3f n = polyasphsurf_normal_vector(P, asurf->center, asurf->c, asurf->k, asurf->poly_coefs, asurf->poly_is_even);
        float t_delta = - e / dot(ray.d, n);

        t += t_delta;
        P = ray(t);
        e = polyasphsurf_implicit_fun(P, asurf->center, asurf->c, asurf->k, asurf->poly_coefs, asurf->poly_is_even);

        bool sel = abs(e) < ae_min;

        ae_min = sel ? abs(e) : ae_min;
        t_min = sel ? t : t_min;

        iter++;
    }

    if(ae_min <= tolerance*10) {
        return;
    }

    if(!point_within_surf_bounds(ray(t_min), asurf->center, asurf->z_min, asurf->z_max, asurf->h_lim)) {
        return;
    }

    optixReportIntersection( t_min, OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE );
}


extern "C" __global__ void __closesthit__poly() {
    const OptixHitGroupData *sbt_data = (OptixHitGroupData *) optixGetSbtDataPointer();
    set_preliminary_intersection_to_payload(
        optixGetRayTmax(), Vector2f(), 0, sbt_data->shape_registry_id);
}
#endif
