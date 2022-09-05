
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>

#if defined(MI_ENABLE_CUDA)
#include "optix/poly.cuh"
#endif

#if 0
#define DEBUG_RAYS
#endif

NAMESPACE_BEGIN(mitsuba)

    template <typename Float, typename Spectrum>
    class PolySurf final : public Shape<Float, Spectrum> {

        using Float3 = Vector<Float, 3>;

        public:
            MI_IMPORT_BASE(Shape, m_to_world, m_to_object, m_is_instance, initialize,
                           mark_dirty, get_children_string, parameters_grad_enabled)
            MI_IMPORT_TYPES()

            using typename Base::ScalarIndex;
            using typename Base::ScalarSize;

        PolySurf(const Properties &props) : Base(props) {
            /// Are the normals pointing inwards relative to the sphere? default: yes for negative curvature, no for positive curvature
            /// This means that the normals are always pointing in the negative z direction by default, i.e. the inside is towards the right halfspace along the z-axis

            m_flip_normals = props.get<bool>("flip_normals", false);

            // Read parameters for polynomial terms
            for(unsigned int n=0; n < NUM_POLY_TERMS; n++) {
                // Even-term polynomial
                if(props.has_property("epoly"+std::to_string(n))) {
                    m_poly_is_even = true;
                    m_poly[n] =
                        props.get<ScalarFloat>("epoly"+std::to_string(n), 0.f);
                }
                // Regular polynomial
                else if(props.has_property("poly"+std::to_string(n))) {
                    m_poly[n] =
                        props.get<ScalarFloat>("poly"+std::to_string(n), 0.f);
                }
                else {
                    m_poly[n] = 0.f;
                }
            }

            // Flip curvature? Can also be specified through negative radius/curvature, note that these combine like -1*-1 = 1

            // Update the to_world transform if center is also provided
            //m_to_world = m_to_world * ScalarTransform4f::translate(props.get<ScalarPoint3f>("center", 0.f));
            //m_to_world = m_to_world * ScalarTransform4f::scale(props.get<ScalarFloat>("radius", 1.f));
            m_to_world =
                m_to_world.scalar() *
                ScalarTransform4f::translate(props.get<ScalarPoint3f>("center", 0.f));

            // h limit is common to both
            m_h_lim = props.get<ScalarFloat>("limit", 0.0f);

            // First lens object - initial parameters
            m_k = props.get<ScalarFloat>("kappa0", 2.f);
            m_r = props.get<ScalarFloat>("radius0", 2.f);
            m_p =  1.0f / m_r.scalar();

            update();

            // PolySurfes' z limit
            m_z_lim = ((dr::sqr(m_h_lim.scalar()) * m_p.scalar()) / (1 + sqrt(1 - (1 + m_k.scalar()) * dr::sqr(m_h_lim.scalar()*m_p.scalar()))));

            // How far into z plane?
            fprintf(stdout, "PolySurf using inv_norm=%s kappa=%.2f radius=%.2f (rho=%f) hlim=%.2f zlim=%.2f\n",
                    m_flip_normals ? "true" : "false",
                    (double) m_k.scalar(), (double) m_r.scalar(), (double) m_p.scalar(),
                    (double) m_h_lim.scalar(), (double) m_z_lim.scalar());

            if( isnan( m_z_lim.scalar() ) ){
                fprintf(stdout, "nan error\n");
                fflush(stdout);
                while(1){};
            }


            initialize();
        }

        void update() {
            // Extract center and radius from to_world matrix (25 iterations for numerical accuracy)
            auto [S, Q, T] = dr::transform_decompose(m_to_world.scalar().matrix, 25);

            if (dr::abs(S[0][1]) > 1e-6f || dr::abs(S[0][2]) > 1e-6f || dr::abs(S[1][0]) > 1e-6f ||
                dr::abs(S[1][2]) > 1e-6f || dr::abs(S[2][0]) > 1e-6f || dr::abs(S[2][1]) > 1e-6f ||
                dr::abs(S[0][0]-1.0f) > 1e-6f || dr::abs(S[1][1]-1.0f) > 1e-6f || dr::abs(S[2][2]-1.0f) > 1e-6f)
                Log(Warn, "'to_world' transform shouldn't contain any scaling or shearing!");

            m_center = ScalarPoint3f(T);

            m_to_world = dr::transform_compose<ScalarMatrix4f>(S, Q, T);
            m_to_object = m_to_world.scalar().inverse();

            m_inv_surface_area = 1.0f;

            dr::make_opaque(m_r, m_center, m_inv_surface_area);
            mark_dirty();
        }


        ScalarBoundingBox3f bbox() const override {
            ScalarBoundingBox3f bbox;

            bbox.min = m_center.scalar() - 1000;
            bbox.max = m_center.scalar() + 1000;

            return bbox;
        }

        Float surface_area() const override {
            return 1.f; // This needs fixing
        }

        // =============================================================
        //! @{ \name Sampling routines
        // =============================================================

        PositionSample3f sample_position(Float time, const Point2f &sample,
                                         Mask active) const override {
            MI_MASK_ARGUMENT(active);

            std::cout << "sample_position\n";
            std::cerr << "sample_position\n";
            std::cerr << sample << "\n";

            Point3f local = warp::square_to_uniform_sphere(sample);

            PositionSample3f ps;
            ps.p = local/* + m_center*/;
            ps.n = local;

            if (m_flip_normals)
                ps.n = -ps.n;

            ps.time = time;
            ps.delta = false;
            ps.pdf = m_inv_surface_area;

            return ps;
        }

        Float pdf_position(const PositionSample3f & /*ps*/, Mask active) const override {
            MI_MASK_ARGUMENT(active);
            std::cerr << "pdf_position\n";
            std::cout << "pdf_position\n";
            return m_inv_surface_area;
        }

        DirectionSample3f sample_direction(const Interaction3f &it, const Point2f &sample,
                                           Mask active) const override {
            MI_MASK_ARGUMENT(active);
            DirectionSample3f result = dr::zeros<DirectionSample3f>();

            (void) it;
            (void) sample;
            (void) active;

            std::cerr << "sample_direction\n";
            std::cout << "sample_direction\n";

            if(1) while(1);

            return result;
        }

        Float pdf_direction(const Interaction3f &it, const DirectionSample3f &ds,
                            Mask active) const override {
            MI_MASK_ARGUMENT(active);
            std::cerr << "pdf_direction\n";
            std::cout << "pdf_direction\n";

            (void) it;
            (void) ds;
            (void) active;

            if(1) while(1);

            Mask ret;

            return ret;
        }


        template <typename Value, typename Ray3fP>
        std::tuple<dr::mask_t<Value>, Value, Value>
        find_intersections( const Ray3fP &ray) const{

            /*
            using Value = std::conditional_t<dr::is_cuda_v<FloatP> ||
                                            dr::is_diff_v<Float>, dr::float32_array_t<FloatP>,
                                            dr::float64_array_t<FloatP>>;
                                            */
            using Value3 = Vector<Value, 3>;
            using ScalarValue  = dr::scalar_t<Value>;
            using ScalarValue3 = Vector<ScalarValue, 3>;

            /*
             * Something from ray_intersect_preliminary_impl
             * */

            Value3 c;
            Value g;
            Value p;
            Value r;

            if constexpr (!dr::is_jit_v<Value>) {
                c = (ScalarValue3) m_center.scalar();
                g = (ScalarValue) ( 1 + m_k.scalar() );
                p = (ScalarValue) m_p.scalar();
                r = (ScalarValue) (m_r.scalar()*2.f);
            } else {
                c = (Value3) m_center.value();
                g = ( 1 + m_k.value() );
                p = m_p.value();
                r = m_r.value()*2.f;
            }

            // Unit vector
            Value3 d(ray.d);

            // Origin
            Value3 o(ray.o);

            Value dx = d[0], dy = d[1], dz = d[2];
            Value ox = o[0], oy = o[1], oz = o[2];

            Value x0 = c[0], y0 = c[1], z0 = c[2];

            Value A = g * dr::sqr(dz) + dr::sqr(dx) + dr::sqr(dy);
            Value B = g * 2 * oz * dz - 2 * g * z0 * dz + 2 * ox * dx - 2 * x0 * dx + 2 * oy * dy - 2 * y0 * dy - 2 * dz / p;
            Value C = g * dr::sqr(oz) - g * 2 * z0 * oz + g * dr::sqr(-1*z0) + dr::sqr(ox) - 2 * x0 * ox + dr::sqr(-1*x0) + dr::sqr(oy) - 2 * y0 * oy + dr::sqr(-1*y0) - 2 * oz / p - 2 * -1*z0 / p;

            Value near, far;

            auto [solution_found, solution0, solution1] = math::solve_quadratic<Value>(A, B, C);

            near = solution0;
            far = solution1;

            return { solution_found, near, far };
        }

        template <typename FloatP, typename Value, typename Value3, typename Ray3fP>
        dr::mask_t<FloatP>
        point_on_lens_surface( Value3 t0 ) const {

            using ScalarValue  = dr::scalar_t<Value>;
            using ScalarValue3 = Vector<ScalarValue, 3>;

            /*
             * Something from ray_intersect_preliminary_impl
             * */
            Value3 center;
            Value z_lim;
            Value h_lim;

            if constexpr (!dr::is_jit_v<Value>) {
                center = (ScalarValue3) m_center.scalar();
                z_lim = (ScalarValue) m_z_lim.scalar();
                h_lim = (ScalarValue) m_h_lim.scalar();
            } else {
                center = (Value3) m_center.value();
                z_lim = m_z_lim.value();
                h_lim = m_h_lim.value();
            }

            //return z_lim < 1.0f;
            Value3 delta0;
            Value hyp0;

            delta0 = t0 - center;

            hyp0 = dr::sqrt( dr::sqr( delta0[0] ) + dr::sqr( delta0[1] ) + dr::sqr( delta0[2] ) );

            Value limit;

            Value w = z_lim;

            limit = dr::sqrt( (dr::sqr( h_lim /*(Float) m_h_lim.scalar()*/ )) + dr::sqr( w ) );

            (void) limit;
            return (hyp0 <= limit);
        }

        //! @}
        // =============================================================

        // =============================================================
        //! @{ \name Ray tracing routines
        // =============================================================

        //PreliminaryIntersection3f ray_intersect_preliminary(const Ray3f &ray,
        //                                                    Mask active) const override {
        template <typename FloatP, typename Ray3fP>
            std::tuple<FloatP, Point<FloatP, 2>, dr::uint32_array_t<FloatP>,
            dr::uint32_array_t<FloatP>>
                ray_intersect_preliminary_impl(const Ray3fP &ray,
                                               dr::mask_t<FloatP> active) const {
                    MI_MASK_ARGUMENT(active);

                    using Value = std::conditional_t<dr::is_cuda_v<FloatP> ||
                                                          dr::is_diff_v<Float>,
                                                      dr::float32_array_t<FloatP>,
                                                      dr::float64_array_t<FloatP>>;
                    using Value3 = Vector<Value, 3>;
                    using ScalarValue  = dr::scalar_t<Value>;
                    using ScalarValue3 = Vector<ScalarValue, 3>;

                    Value maxt = Value(ray.maxt);

                    auto [ solution, near_t, far_t ] = find_intersections<Value, Ray3fP>( ray);

                    dr::mask_t<FloatP> result0, result1;

                    // Where on the sphere plane is that?
                    result0 = point_on_lens_surface<FloatP, Value, Value3, Ray3fP>( ray(near_t) ) && near_t < maxt;
                    result1 = point_on_lens_surface<FloatP, Value, Value3, Ray3fP>( ray(far_t) ) && far_t < maxt;

                    FloatP t = dr::select( result0 && (near_t > Value(0.0)),
                                           FloatP(near_t),
                                           dr::select( result1 && (far_t > Value(0.0)),
                                                       FloatP(far_t),
                                                       dr::Infinity<FloatP>) );

                    t = dr::select( near_t <= maxt, t, dr::Infinity<FloatP> );

                    return { t, dr::zeros<Point<FloatP, 2>>(), ((uint32_t) -1), 0 };
                }

        //Mask ray_test(const Ray3f &ray, Mask active) const override {
        template <typename FloatP, typename Ray3fP>
            dr::mask_t<FloatP> ray_test_impl(const Ray3fP &ray,
                                             dr::mask_t<FloatP> active) const {
                MI_MASK_ARGUMENT(active);

                dr::mask_t<FloatP> solution_found;

                dr::mask_t<FloatP> active_ = active;

                auto [t, unused0, unused1, unused2] = ray_intersect_preliminary_impl<FloatP, Ray3fP>(ray, active_);

                solution_found = (t != dr::Infinity<Float>);

                return solution_found && /* !out_bounds && !in_bounds && */ active;
            }

        MI_SHAPE_DEFINE_RAY_INTERSECT_METHODS()

        SurfaceInteraction3f compute_surface_interaction(const Ray3f &ray,
                                                         const PreliminaryIntersection3f &pi,
                                                         uint32_t ray_flags,
                                                         uint32_t recursion_depth,
                                                         Mask active) const override {
            MI_MASK_ARGUMENT(active);

            // Early exit when tracing isn't necessary
            if (!m_is_instance && recursion_depth > 0)
                return dr::zeros<SurfaceInteraction3f>();

            // Recompute ray intersection to get differentiable t
            Float t = pi.t;
            if constexpr (dr::is_diff_v<Float>)
                t = dr::replace_grad(t, ray_intersect_preliminary(ray, active).t);

            // TODO handle RayFlags::FollowShape and RayFlags::DetachShape

            // Fields requirement dependencies
            bool need_dn_duv = has_flag(ray_flags, RayFlags::dNSdUV) ||
                               has_flag(ray_flags, RayFlags::dNGdUV);
            bool need_dp_duv = has_flag(ray_flags, RayFlags::dPdUV) || need_dn_duv;
            bool need_uv     = has_flag(ray_flags, RayFlags::UV) || need_dp_duv;

            active &= pi.is_valid();

            SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
            si.t = dr::select(active, t, dr::Infinity<Float>);

            Float3 point = ray(pi.t) - m_center.scalar();

            /*
             * Now compute the unit vector
             * */
            Float fx, fy, fz;
            Float p = m_p.scalar();
            Float k(m_k.scalar());

            fx = ( point[0] * p ) / dr::sqrt( 1 - (1+k) * (dr::sqr(point[0]) + dr::sqr(point[1])) * dr::sqr(p));
            fy = ( point[1] * p ) / dr::sqrt( 1 - (1+k) * (dr::sqr(point[0]) + dr::sqr(point[1])) * dr::sqr(p));
            fz = -1.0;

            if( ! m_flip_normals )
                si.sh_frame.n = normalize( Float3( fx, fy, fz ) );
            else
                si.sh_frame.n = normalize( Float(-1) * Float3( fx, fy, fz ) );

            // Frame.n is a unit vector. between the center of the
            // ellipsis and the crossing point apparently.
            si.p = ray(pi.t);

            if (likely(need_uv)) {
                Vector3f local = m_to_object.value().transform_affine(si.p);

                si.uv = Point2f( local.x() / m_r.scalar(),
                                 local.y() / m_r.scalar() );

                //if (likely(has_flag(flags, HitComputeFlags::dPdUV))) {
                if (likely(need_dp_duv)) {

                    si.dp_du = Vector3f( fx, 1.0, 0.0 );
                    si.dp_dv = Vector3f( fy, 0.0, 1.0 );
                }

            }

            si.n = si.sh_frame.n;

            if (need_dn_duv) {
                Float inv_radius =
                    (m_flip_normals ? -1.f : 1.f) * dr::rcp(m_r.value());
                si.dn_du = si.dp_du * inv_radius;
                si.dn_dv = si.dp_dv * inv_radius;
            }

            si.shape    = this;
            si.instance = nullptr;

            if (unlikely(has_flag(ray_flags, RayFlags::BoundaryTest)))
                si.boundary_test = dr::abs(dr::dot(si.sh_frame.n, -ray.d));

            if (need_dn_duv) {
                std::cout << "dNSdUV\n";
                Log(Warn, "dNSdUV");
                Log(Error, "dNSdUV");
                if(1) while(1);
            }

            return si;
        }

        //! @}
        // =============================================================

        void traverse(TraversalCallback *callback) override {
            std::cerr << "traverse\n";
            std::cout << "traverse\n";
            Base::traverse(callback);
        }

        void parameters_changed(const std::vector<std::string> &keys) override {
            std::cerr << "parameters_changed\n";
            std::cout << "parameters_changed\n";

            if (keys.empty() || string::contains(keys, "to_world")) {
                // Update the scalar value of the matrix
                m_to_world = m_to_world.value();
                update();
            }
            Base::parameters_changed();
        }

#if defined(MI_ENABLE_CUDA)
        using Base::m_optix_data_ptr;

        void optix_prepare_geometry() override {
            if constexpr (dr::is_cuda_v<Float>) {
                if (!m_optix_data_ptr)
                    m_optix_data_ptr = jit_malloc(AllocType::Device, sizeof(OptixPolyData));

                OptixPolyData data = { bbox(),
                    (Vector<float, 3>) m_center.scalar(),
                    (float) m_r.scalar(),
                    (float) m_k.scalar(),
                    (float) m_p.scalar(),
                    (float) m_r.scalar(),
                    (float) m_h_lim.scalar(),
                    (float) m_z_lim.scalar(),
                    m_flip_normals };

                jit_memcpy(JitBackend::CUDA, m_optix_data_ptr,
                           &data, sizeof(OptixPolyData));
            }
        }
#endif

        std::string to_string() const override {
            std::ostringstream oss;
            oss << "PolySurf[" << std::endl
                << "  to_world = " << string::indent(m_to_world, 13) << "," << std::endl
                << "  center = "  << m_center << "," << std::endl
                << "  radius = "  << m_r << "," << std::endl
                << "  surface_area = " << surface_area() << "," << std::endl
                << "  " << string::indent(get_children_string()) << std::endl
                << "]";
            return oss.str();
        }

        MI_DECLARE_CLASS()
        private:
            /// Center in world-space
            field<Point3f> m_center;
            /// Polynomial
            Vector<ScalarFloat, NUM_POLY_TERMS> m_poly;
            bool m_poly_is_even = false;
            /// kappa
            field<Float> m_k;
            /// curvature
            field<Float> m_p;
            /// radius
            field<Float> m_r;

            /// limit of h
            field<Float> m_h_lim;

            /// how far into the "z plane" the surface reaches
            /// -- it is a function of m_h_lim
            field<Float> m_z_lim;

            Float m_inv_surface_area;

            bool m_flip_normals;
    };

    MI_IMPLEMENT_CLASS_VARIANT(PolySurf, Shape)
    MI_EXPORT_PLUGIN(PolySurf, "PolySurf intersection primitive");
    NAMESPACE_END(mitsuba)
