
#pragma once

#include <math.h>
#include <mitsuba/render/optix/common.h>
#include <mitsuba/render/optix/math.cuh>

struct OptixConicData {
    optix::BoundingBox3f bbox;
    optix::Vector3f center;

    float radius;

    float k;
    float p;
    float r;
    float h_lim;
    float z_lim;

    bool flip_normals;
};

#ifdef __CUDACC__

bool __device__ point_on_lens_surface( Vector3f point, Vector3f center, float z_lim, float h_lim) {

    Vector3f delta0;
    float hyp0;

    delta0 = point - center;

    hyp0 = sqrt( sqr( delta0[0] ) + sqr( delta0[1] ) + sqr( delta0[2] ) );

    float limit;

    float w = (float) z_lim;

    limit = sqrt( (sqr( (float) h_lim )) + sqr( w ) );

    return (hyp0 <= limit);
}

bool __device__ find_intersections0( float &near_t, float &far_t,
                          Vector3f center,
                          float m_p, float m_k,
                          const Ray3f &ray){

    // Unit vector
    Vector3f d = ray.d;

    // Origin
    Vector3f o = ray.o;

    // Center of sphere
    Vector3f c = center;

    float dx = d[0], dy = d[1], dz = d[2];
    float ox = o[0], oy = o[1], oz = o[2];

    float x0 = c[0], y0 = c[1], z0 = c[2];

    float g = ( 1 + m_k );

    float A = g * sqr(dz) + sqr(dx) + sqr(dy);
    float B = g * 2 * oz * dz - 2 * g * z0 * dz + 2 * ox * dx - 2 * x0 * dx + 2 * oy * dy - 2 * y0 * dy - 2 * dz / m_p;
    float C = g * sqr(oz) - g * 2 * z0 * oz + g * sqr(-1*z0) + sqr(ox) - 2 * x0 * ox + sqr(-1*x0) + sqr(oy) - 2 * y0 * oy + sqr(-1*y0) - 2 * oz / m_p - 2 * -1*z0 / m_p;

    bool solution_found = solve_quadratic(A, B, C, near_t, far_t);

    return solution_found;
}


extern "C" __global__ void __intersection__conic() {
    const OptixHitGroupData *sbt_data = (OptixHitGroupData*) optixGetSbtDataPointer();

    OptixConicData *asurf = (OptixConicData *)sbt_data->data;

    // Ray in instance-space
    Ray3f ray = get_ray();

    float near_t0, far_t0;

    bool solution;
    bool valid0, valid1;

#if 0
    if( /*asurf->flip*/ asurf->p < 0 ){
        solution0 = find_intersections0( near_t0, far_t0,
                                         asurf->center - Vector3f(0,0, asurf->r * 2.f),
                                         asurf->p, asurf->k,
                                         ray);

        near_t0 = far_t0; // hack hack
    }
    else{
#else
        solution = find_intersections0( near_t0, far_t0,
                                         asurf->center,
                                         asurf->p, asurf->k,
                                         ray);
#endif
    //}

    if( ! solution ){
        return;
    }

    // Where on the sphere plane is that?
    valid0 = point_on_lens_surface( ray(near_t0),
                    asurf->center, 
                    asurf->z_lim, asurf->h_lim );
    valid1 = point_on_lens_surface( ray(far_t0),
                    asurf->center,
                    asurf->z_lim, asurf->h_lim );

    // Need to incorporate the in_bounds and out_bounds checking here.

    if( near_t0 > 0.0 && valid0){
        optixReportIntersection( near_t0, OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE );
    }
    else if( far_t0 > 0.0 && valid1 ){
        optixReportIntersection( far_t0, OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE );
    }
    else{
        //PASS
    }
}

/*
 * This function is much simpler now than in mitsuba2.
 * So try to just leave it like this.
 * */
extern "C" __global__ void __closesthit__conic() {
    const OptixHitGroupData *sbt_data = (OptixHitGroupData *) optixGetSbtDataPointer();
    set_preliminary_intersection_to_payload(
        optixGetRayTmax(), Vector2f(), 0, sbt_data->shape_registry_id);
}

#endif
