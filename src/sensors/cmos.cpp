
#include <mitsuba/render/sensor.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/bbox.h>
#include <mitsuba/core/warp.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sensor-thinlens:

Perspective camera with a thin lens (:monosp:`thinlens`)
--------------------------------------------------------

.. pluginparameters::

 * - to_world
   - |transform|
   - Specifies an optional camera-to-world transformation.
     (Default: none (i.e. camera space = world space))
 * - aperture_radius
   - |float|
   - Denotes the radius of the camera's aperture in scene units.
 * - focus_distance
   - |float|
   - Denotes the world-space distance from the camera's aperture to the focal plane.
     (Default: :monosp:`0`)
 * - focal_length (unused)
   - |string|
   - Denotes the camera's focal length specified using *35mm* film equivalent units.
     See the main description for further details. (Default: :monosp:`50mm`)
 * - fov (unused)
   - |float|
   - An alternative to :monosp:`focal_length`: denotes the camera's field of view in degrees---must be
     between 0 and 180, excluding the extremes.
 * - fov_axis (unused)
   - |string|
   - When the parameter :monosp:`fov` is given (and only then), this parameter further specifies
     the image axis, to which it applies.

     1. :monosp:`x`: :monosp:`fov` maps to the :monosp:`x`-axis in screen space.
     2. :monosp:`y`: :monosp:`fov` maps to the :monosp:`y`-axis in screen space.
     3. :monosp:`diagonal`: :monosp:`fov` maps to the screen diagonal.
     4. :monosp:`smaller`: :monosp:`fov` maps to the smaller dimension
        (e.g. :monosp:`x` when :monosp:`width` < :monosp:`height`)
     5. :monosp:`larger`: :monosp:`fov` maps to the larger dimension
        (e.g. :monosp:`y` when :monosp:`width` < :monosp:`height`)

     The default is :monosp:`x`.
 * - near_clip, far_clip
   - |float|
   - Distance to the near/far clip planes. (Default: :monosp:`near_clip=1e-2` (i.e. :monosp:`0.01`)
     and :monosp:`far_clip=1e4` (i.e. :monosp:`10000`))

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/sensor_thinlens_small_aperture.jpg
   :caption: The material test ball viewed through a perspective thin lens camera. (:monosp:`aperture_radius=0.1`)
.. subfigure:: ../../resources/data/docs/images/render/sensor_thinlens.jpg
   :caption: The material test ball viewed through a perspective thin lens camera. (:monosp:`aperture_radius=0.2`)
.. subfigend::
   :label: fig-thinlens

This plugin implements a simple perspective camera model with a thin lens
at its circular aperture. It is very similar to the
:ref:`perspective <sensor-perspective>` plugin except that the extra lens element
permits rendering with a specifiable (i.e. non-infinite) depth of field.
To configure this, it has two extra parameters named :monosp:`aperture_radius`
and :monosp:`focus_distance`.

By default, the camera's field of view is specified using a 35mm film
equivalent focal length, which is first converted into a diagonal field
of view and subsequently applied to the camera. This assumes that
the film's aspect ratio matches that of 35mm film (1.5:1), though the
parameter still behaves intuitively when this is not the case.
Alternatively, it is also possible to specify a field of view in degrees
along a given axis (see the :monosp:`fov` and :monosp:`fov_axis` parameters).

The exact camera position and orientation is most easily expressed using the
:monosp:`lookat` tag, i.e.:

.. code-block:: xml

    <sensor type="thinlens">
        <transform name="to_world">
            <!-- Move and rotate the camera so that looks from (1, 1, 1) to (1, 2, 1)
                and the direction (0, 0, 1) points "up" in the output image -->
            <lookat origin="1, 1, 1" target="1, 2, 1" up="0, 0, 1"/>
        </transform>

        <!-- Focus on the target -->
        <float name="focus_distance" value="1"/>
        <float name="aperture_radius" value="0.1"/>
    </sensor>

 */

static int dbg = 0;

template <typename Float, typename Spectrum>
class CmosCamera final : public ProjectiveCamera<Float, Spectrum> {
public:
    MI_IMPORT_BASE(ProjectiveCamera, m_to_world, m_needs_sample_3, m_film, m_sampler,
                    m_resolution, m_shutter_open, m_shutter_open_time, m_near_clip,
                    m_far_clip, m_focus_distance, sample_wavelengths)
    MI_IMPORT_TYPES()

    // =============================================================
    //! @{ \name Constructors
    // =============================================================

    CmosCamera(const Properties &props) : Base(props) {

        ScalarVector2i size = m_film->size();
        m_x_fov = parse_fov(props, size.x() / (double) size.y());

        m_aperture_radius = props.get<ScalarFloat>("aperture_radius");

        if (m_aperture_radius == 0.f) {
            Log(Warn, "Can't have a zero aperture radius -- setting to %f", drjit::Epsilon<Float>);
            m_aperture_radius = drjit::Epsilon<Float>;
        }

        if (m_to_world.scalar().has_scale())
            Throw("Scale factors in the camera-to-world transformation are not allowed!");

        m_width = props.get<ScalarFloat>("sensor_width", -1.f);
        m_height = props.get<ScalarFloat>("sensor_height", -1.f);

        m_cw = props.get<ScalarFloat>("cw", -1.f);
        m_ch = props.get<ScalarFloat>("ch", -1.f);

#if 0 // FIX THIS
        if( ((m_width > 0.f) && (m_cw > 0.f)) ||
            ((m_height > 0.f) && (m_ch > 0.f))){
            Log(Error, "Specify total die size using 'width' "
                        "or 'height' OR the pixel cell size "
                        "using 'cw' and 'ch', not both." );
        }

        if( m_cw > 0.f ){
            m_width = m_resolution.x() * m_cw / 1.0E3f;
        }
        if( m_ch > 0.f ){
            m_height = m_resolution.y() * m_ch / 1.0E3f;
        }
#endif

        m_camera_to_sample = perspective_projection(
            m_film->size(), m_film->crop_size(), m_film->crop_offset(),
            m_x_fov, Float(m_near_clip), Float(m_far_clip));

        m_sample_to_camera = m_camera_to_sample.inverse();

        // Position differentials on the near plane
        m_dx = m_sample_to_camera * Point3f(1.f / m_resolution.x(), 0.f, 0.f)
             - m_sample_to_camera * Point3f(0.f);
        m_dy = m_sample_to_camera * Point3f(0.f, 1.f / m_resolution.y(), 0.f)
             - m_sample_to_camera * Point3f(0.f);

        /* Precompute some data for importance(). Please
           look at that function for further details. */
        Point3f pmin(m_sample_to_camera * Point3f(0.f, 0.f, 0.f)),
                      pmax(m_sample_to_camera * Point3f(1.f, 1.f, 0.f));

        m_image_rect.reset();
        m_image_rect.expand(Point2f(pmin.x(), pmin.y()) / pmin.z());
        m_image_rect.expand(Point2f(pmax.x(), pmax.y()) / pmax.z());
        m_normalization = 1.f / m_image_rect.volume();
        m_needs_sample_3 = true;
    }

    //! @}
    // =============================================================

    // =============================================================
    //! @{ \name Sampling methods (Sensor interface)
    // =============================================================

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &position_sample,
                                          const Point2f &aperture_sample,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

#if 1
        auto [wavelengths, wav_weight] = sample_wavelength<Float, Spectrum>(wavelength_sample);
        Ray3f ray;
        ray.time = time;
        ray.wavelengths = wavelengths;

        Point3f focus_p = Point3f( (position_sample.x() -.5f) * m_width,
                                   (position_sample.y() -.5f) * m_height, 0.f );

        // Aperture position
        Point2f tmp = m_aperture_radius * warp::square_to_uniform_disk_concentric(aperture_sample);

        Point3f aperture_p(tmp.x(), tmp.y(), m_focus_distance);

#if 0
        if( 0 || ++dbg > 10){
            std::cerr << "point0," << focus_p[0] << "," << focus_p[1] << "," << focus_p[2] << std::endl;
            std::cerr << "point1," << aperture_p[0] << "," << aperture_p[1] << "," << aperture_p[2] << std::endl;
            dbg = 0;
        }
#endif

        Vector3f d = normalize(Vector3f(aperture_p - focus_p));
        Float inv_z = dr::rcp(d.z());
        //ray.mint = m_near_clip * inv_z;
        ray.maxt = m_far_clip * inv_z;

        //ray.o = m_to_world.value().transform_affine(aperture_p);
        //auto trafo = m_to_world->eval(ray.time, active);

        ray.o = m_to_world.value().transform_affine(focus_p);
        ray.d = m_to_world.value() * d;
        //ray.update();

#if 0
        if( 1 || ++dbg > 10){
            std::cerr << "point0," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << std::endl;
            //std::cerr << "vec0," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2] << std::endl;
            dbg = 0;
        }
#endif

#else
        auto [wavelengths, wav_weight] =
            sample_wavelengths(dr::zeros<SurfaceInteraction3f>(),
                               wavelength_sample,
                               active);
#warning "Fix origin ray and sanity check in constructor"
        Ray3f ray;
        ray.time = time;
        ray.wavelengths = wavelengths;

        // Compute the sample position on the near plane (local camera space).
        Point3f near_p = m_sample_to_camera *
                        Point3f(position_sample.x(), position_sample.y(), 0.f);
        Point3f focus_p = Point3f( (position_sample.x() -.5f) * m_width,
                                   (position_sample.y() -.5f) * m_height, 0.f );


        // Aperture position
        Point2f tmp = m_aperture_radius * warp::square_to_uniform_disk_concentric(aperture_sample);
        Point3f aperture_p(tmp.x(), tmp.y(), 0.f);

        // Sampled position on the focal plane
        Point3f focus_p = near_p * (m_focus_distance / near_p.z());

        if( 0 || ++dbg > 50000){
            std::cerr << "point0," << near_p[0] << "," << near_p[1] << "," << near_p[2] << std::endl;
            std::cerr << "point1," << aperture_p[0] << "," << aperture_p[1] << "," << aperture_p[2] << std::endl;
            dbg = 0;
        }
#if 0
        std::cout << "near_p.z() " << near_p.z() << std::endl;
        std::cout << m_focus_distance << std::endl;
        std::cout << "aperture_p " << aperture_p << std::endl;
        std::cout << "focus_p " << focus_p << std::endl;
#endif

        // Convert into a normalized ray direction; adjust the ray interval accordingly.
        Vector3f d = dr::normalize(Vector3f(focus_p - aperture_p));

        //ray.o = m_to_world.value().transform_affine(aperture_p);
        //std::cout << "ray.o " << ray.o << std::endl;
        ray.o = m_to_world.value().transform_affine(focus_p);
        ray.o -= Point3f(0,0,47);
        //std::cout << "ray.o " << ray.o << std::endl;
        ray.d = m_to_world.value() * d;

        Float inv_z = dr::rcp(d.z());
        Float near_t = m_near_clip * inv_z,
              far_t  = m_far_clip * inv_z;
        ray.o += ray.d * near_t;
        ray.maxt = far_t - near_t;

#if 0
        if( 0 || ++dbg > 50000){
            std::cerr << "point0," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << std::endl;
            std::cerr << "vec0," << ray.o[0] << "," << ray.o[1] << "," << ray.o[2] << "," << ray.d[0] << "," << ray.d[1] << "," << ray.d[2] << std::endl;
            dbg = 0;
        }
#endif
#endif

        return std::make_pair(ray, wav_weight);
    }

    std::pair<RayDifferential3f, Spectrum>
    sample_ray_differential_impl(Float time, Float wavelength_sample,
                                 const Point2f &position_sample, const Point2f &aperture_sample,
                                 Mask active) const {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        std::cout << "NOT IMPLEMENTED!" << std::endl;
        std::cout << "NOT IMPLEMENTED!" << std::endl;
        std::cout << "NOT IMPLEMENTED!" << std::endl;
        std::cout << "NOT IMPLEMENTED!" << std::endl;
        std::cout << "NOT IMPLEMENTED!" << std::endl;
        while(1){};

        auto [wavelengths, wav_weight] =
            sample_wavelengths(dr::zeros<SurfaceInteraction3f>(),
                               wavelength_sample,
                               active);
        RayDifferential3f ray;
        ray.time = time;
        ray.wavelengths = wavelengths;

        // Compute the sample position on the near plane (local camera space).
        Point3f near_p = m_sample_to_camera *
                        Point3f(position_sample.x(), position_sample.y(), 0.f);

        // Aperture position
        Point2f tmp = m_aperture_radius * warp::square_to_uniform_disk_concentric(aperture_sample);
        Point3f aperture_p(tmp.x(), tmp.y(), 0.f);

        // Sampled position on the focal plane
        Float f_dist = m_focus_distance / near_p.z();
        Point3f focus_p   = near_p          * f_dist,
                focus_p_x = (near_p + m_dx) * f_dist,
                focus_p_y = (near_p + m_dy) * f_dist;

        // Convert into a normalized ray direction; adjust the ray interval accordingly.
        Vector3f d = dr::normalize(Vector3f(focus_p - aperture_p));

        ray.o = m_to_world.value().transform_affine(aperture_p);
        ray.d = m_to_world.value() * d;

        Float inv_z = dr::rcp(d.z());
        Float near_t = m_near_clip * inv_z,
              far_t  = m_far_clip * inv_z;
        ray.o += ray.d * near_t;
        ray.maxt = far_t - near_t;

        ray.o_x = ray.o_y = ray.o;

        ray.d_x = m_to_world.value() * dr::normalize(Vector3f(focus_p_x - aperture_p));
        ray.d_y = m_to_world.value() * dr::normalize(Vector3f(focus_p_y - aperture_p));
        ray.has_differentials = true;

        return { ray, wav_weight };
    }

    ScalarBoundingBox3f bbox() const override {
        ScalarPoint3f p = m_to_world.scalar() * ScalarPoint3f(0.f);
        return ScalarBoundingBox3f(p, p);
    }

    //! @}
    // =============================================================

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        // TODO aperture_radius, x_fov
    }

    void parameters_changed(const std::vector<std::string> &keys) override {
        Base::parameters_changed(keys);
        // TODO
    }

    std::string to_string() const override {
        using string::indent;

        std::ostringstream oss;
        oss << "CmosCamera[" << std::endl
            << "  x_fov = " << m_x_fov << "," << std::endl
            << "  near_clip = " << m_near_clip << "," << std::endl
            << "  far_clip = " << m_far_clip << "," << std::endl
            << "  focus_distance = " << m_focus_distance << "," << std::endl
            << "  film = " << indent(m_film) << "," << std::endl
            << "  sampler = " << indent(m_sampler) << "," << std::endl
            << "  resolution = " << m_resolution << "," << std::endl
            << "  shutter_open = " << m_shutter_open << "," << std::endl
            << "  shutter_open_time = " << m_shutter_open_time << "," << std::endl
            << "  world_transform = " << indent(m_to_world)  << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    Transform4f m_camera_to_sample;
    Transform4f m_sample_to_camera;
    BoundingBox2f m_image_rect;
    Float m_aperture_radius;
    Float m_normalization;
    Float m_x_fov;
    Vector3f m_dx, m_dy;

    /// New stuff
    Float m_width;
    Float m_height;
    Float m_cw;
    Float m_ch;

    //ScalarTransform4f m_test_transform;
};

MI_IMPLEMENT_CLASS_VARIANT(CmosCamera, ProjectiveCamera)
MI_EXPORT_PLUGIN(CmosCamera, "CMOS Camera");
NAMESPACE_END(mitsuba)
