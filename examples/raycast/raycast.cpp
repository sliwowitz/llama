#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <stb_image_write.h>
#include <stdexcept>
#include <string>

namespace
{
    constexpr auto pi = 3.14159265359f;

    template <typename F>
    struct Vector
    {
        auto operator[](std::size_t index) -> F&
        {
            return values[index];
        }

        auto operator[](std::size_t index) const -> F
        {
            return values[index];
        }

        auto operator+=(const Vector& v) -> Vector&
        {
            for (int i = 0; i < 3; i++)
                values[i] += v.values[i];
            return *this;
        }

        auto operator-=(const Vector& v) -> Vector&
        {
            for (int i = 0; i < 3; i++)
                values[i] -= v.values[i];
            return *this;
        }

        template <typename Scalar>
        auto operator*=(Scalar scalar) -> Vector&
        {
            for (int i = 0; i < 3; i++)
                values[i] *= scalar;
            return *this;
        }

        auto lengthSqr() const
        {
            F r = 0;
            for (int i = 0; i < 3; i++)
                r += values[i] * values[i];
            return r;
        }

        auto length() const
        {
            return std::sqrt(lengthSqr());
        }

        void normalize()
        {
            const auto l = length();
            for (int i = 0; i < 3; i++)
                values[i] /= l;
        }

        auto normalized() const -> Vector
        {
            auto r = *this;
            r.normalize();
            return r;
        }

        friend auto operator+(const Vector& a, const Vector& b) -> Vector
        {
            auto r = a;
            r += b;
            return r;
        }

        friend auto operator-(const Vector& a, const Vector& b) -> Vector
        {
            auto r = a;
            r -= b;
            return r;
        }

        friend auto operator*(const Vector& v, F scalar) -> Vector
        {
            auto r = v;
            r *= scalar;
            return r;
        }

        friend auto operator*(F scalar, const Vector& v) -> Vector
        {
            return v * scalar;
        }

        friend auto operator>>(std::istream& is, Vector& v) -> std::istream&
        {
            for (int i = 0; i < 3; i++)
                is >> v[i];
            return is;
        }

        std::array<F, 3> values = {{0, 0, 0}};
    };

    template <typename F>
    inline auto dot(const Vector<F>& a, const Vector<F>& b) -> F
    {
        F r = 0;
        for (int i = 0; i < 3; i++)
            r += a[i] * b[i];
        return r;
    }

    template <typename F>
    inline auto cross(const Vector<F>& a, const Vector<F>& b) -> Vector<F>
    {
        Vector<F> r;
        r[0] = a[1] * b[2] - a[2] * b[1];
        r[1] = a[2] * b[0] - a[0] * b[2];
        r[2] = a[0] * b[1] - a[1] * b[0];
        return r;
    }

    using VectorF = Vector<float>;

    inline auto solveQuadraticEquation(double a, double b, double c) -> std::vector<double>
    {
        const double discriminat = b * b - 4 * a * c;
        if (discriminat < 0)
            return {};

        if (discriminat == 0)
            return {-b / 2 * a};

        const auto x1 = (-b - std::sqrt(discriminat)) / 2 * a;
        const auto x2 = (-b + std::sqrt(discriminat)) / 2 * a;
        return {x1, x2};
    }

    struct Camera
    {
        float fovy; // in degree
        VectorF position;
        VectorF view;
        VectorF up;
    };

    struct Sphere
    {
        VectorF center;
        float radius;
    };

    struct Scene
    {
        Camera camera;
        std::vector<Sphere> spheres;
    };

    class Image
    {
    public:
        using Pixel = Vector<unsigned char>;

        Image(unsigned int width, unsigned int height) : width(width), height(height), pixels(width * height)
        {
        }

        auto operator()(unsigned int x, unsigned int y) -> Pixel&
        {
            return pixels[y * width + x];
        }

        auto operator()(unsigned int x, unsigned int y) const -> const Pixel&
        {
            return pixels[y * width + x];
        }

        void write(const std::filesystem::path& filename) const
        {
            if (!stbi_write_png(filename.string().c_str(), width, height, 3, pixels.data(), 0))
                throw std::runtime_error("Failed to write image " + filename.string());
        }

    private:
        unsigned int width;
        unsigned int height;
        std::vector<Pixel> pixels;
    };

    struct Intersection
    {
        float distance;
        VectorF point;
        VectorF normal;
    };

    struct Ray
    {
        VectorF origin;
        VectorF direction;
    };

    auto createRay(const Camera& camera, unsigned int width, unsigned int height, unsigned int x, unsigned int y) -> Ray
    {
        // we imagine a plane with the image just 1 before the camera, and then we shoot at those pixels

        const auto center = camera.position + camera.view;
        const auto xVec = cross(camera.view, camera.up);
        const auto yVec = camera.up;

        const auto delta = (std::tan(camera.fovy * pi / 180.0f) * 2) / (height - 1);
        const auto xDeltaVec = xVec * delta;
        const auto yDeltaVec = yVec * delta;

        const auto xRel = (x - static_cast<float>(width - 1) / 2);
        const auto yRel = (y - static_cast<float>(height - 1) / 2);

        const auto pixel = center + xDeltaVec * xRel + yDeltaVec * yRel;

        Ray r;
        r.origin = center;
        r.direction = (pixel - camera.position).normalized();
        return r;
    }

    auto intersect(const Ray& ray, const Sphere& sphere) -> std::optional<Intersection>
    {
        // from
        // https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection

        // solve quadratic equation
        const auto a = 1;
        const auto b = 2 * dot(ray.direction, (ray.origin - sphere.center));
        const auto c = (ray.origin - sphere.center).lengthSqr() - sphere.radius * sphere.radius;

        const auto solutions = solveQuadraticEquation(a, b, c);
        if (solutions.empty())
            return {};

        // report the closer intersection
        const auto t = static_cast<float>(*std::min_element(std::begin(solutions), std::end(solutions)));

        Intersection inter;
        inter.distance = t;
        inter.point = ray.origin + t * ray.direction;
        inter.normal = (inter.point - sphere.center).normalized();
        return inter;
    }

    auto colorByRay(const Ray& ray) -> Image::Pixel
    {
        Image::Pixel c;
        for (int i = 0; i < 3; i++)
            c[i] = static_cast<unsigned char>(std::abs(ray.direction[i]) * 255);
        return c;
    }

    auto colorByNearestIntersectionNormal(const std::vector<Intersection>& hits) -> Image::Pixel
    {
        const auto it = std::min_element(std::begin(hits), std::end(hits), [](const auto& a, const auto& b) {
            return a.distance < b.distance;
        });

        if (it == std::end(hits))
        {
            // no hit, black color
            return {};
        }
        else
        {
            Image::Pixel r;
            for (int i = 0; i < 3; i++)
                r[i] = static_cast<unsigned char>(std::abs(it->normal[i]) * 255);
            return r;
        }
    }

    auto colorByIntersectionNormal(std::vector<Intersection> hits) -> Image::Pixel
    {
        constexpr auto translucency = 0.5f;

        std::sort(std::begin(hits), std::end(hits), [](const auto& a, const auto& b) {
            return a.distance < b.distance;
        });

        auto t = translucency;
        Image::Pixel r;
        for (const auto& hit : hits)
        {
            // each hit contributes to the color, lesser with each iteration
            for (int i = 0; i < 3; i++)
                r[i] += static_cast<unsigned char>(std::abs(t * hit.normal[i]) * 255);
            t *= translucency;
        }

        for (int i = 0; i < 3; i++)
            r[i] = std::clamp<unsigned char>(r[i], 0, 255);

        return r;
    }

    auto raycast(const Scene& scene, unsigned int width, unsigned int height) -> Image
    {
        Image img(width, height);

        for (auto y = 0u; y < height; y++)
        {
            for (auto x = 0u; x < width; x++)
            {
                const auto ray = createRay(scene.camera, width, height, x, y);

                std::vector<Intersection> hits;
                for (const auto& sphere : scene.spheres)
                    if (const auto hit = intersect(ray, sphere))
                        hits.push_back(*hit);

                // img(x, y) = colorByRay(ray);
                // img(x, y) = colorByNearestIntersectionNormal(hits);
                img(x, y) = colorByIntersectionNormal(hits);
            }
        }

        return img;
    }

    auto lookAt(float fovy, VectorF pos, VectorF lookAt, VectorF up) -> Camera
    {
        const auto view = (lookAt - pos).normalized();
        const auto up2 = cross(view, cross(view, up)).normalized();
        return Camera{fovy, pos, view, up};
    }

    auto cubicBallsScene() -> Scene
    {
        const auto camera = lookAt(45, {5, 5.5, 6}, {0, 0, 0}, {0, 1, 0});
        auto spheres = std::vector<Sphere>{};
        for (auto z = -2; z <= 2; z++)
            for (auto y = -2; y <= 2; y++)
                for (auto x = -2; x <= 2; x++)
                    spheres.push_back(Sphere{{(float) x, (float) y, (float) z}, 0.8f});
        return Scene{camera, std::move(spheres)};
    }

    auto axisBallsScene() -> Scene
    {
        const auto camera = lookAt(45, {5, 5, 10}, {0, 0, 0}, {0, 1, 0});
        auto spheres = std::vector<Sphere>{
            {{0, 0, 0}, 3.0f},
            {{0, 0, 5}, 2.0f},
            {{0, 5, 0}, 2.0f},
            {{5, 0, 0}, 2.0f},
            {{0, 0, -5}, 1.0f},
            {{0, -5, 0}, 1.0f},
            {{-5, 0, 0}, 1.0f}};
        return Scene{camera, std::move(spheres)};
    }

    auto randomSphereScene() -> Scene
    {
        constexpr auto count = 1024;

        const auto camera = lookAt(45, {5, 5.5, 6}, {0, 0, 0}, {0, 1, 0});
        auto spheres = std::vector<Sphere>{};

        std::default_random_engine eng;
        std::uniform_real_distribution d{-2.0f, 2.0f};
        for (auto i = 0; i < count; i++)
            spheres.push_back({{d(eng), d(eng), d(eng)}, 0.2f});
        return Scene{camera, std::move(spheres)};
    }
} // namespace

int main(int argc, const char* argv[])
try
{
    const auto width = 800;
    const auto height = 600;

    // const auto scene = loadScene(sceneFile);
    // const auto scene = cubicBallsScene();
    // const auto scene = axisBallsScene();
    const auto scene = randomSphereScene();

    const auto start = std::chrono::high_resolution_clock::now();
    const auto image = raycast(scene, width, height);
    const auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Raycast took " << std::chrono::duration<double>(end - start).count() << "s\n";

    image.write("out.png");
    std::system("out.png");
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
    return 2;
}
