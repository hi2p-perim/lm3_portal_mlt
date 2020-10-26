/*
    Lightmetrica - Copyright (c) 2019 Hisanari Otsu
    Distributed under MIT license. See LICENSE file for details.
*/

#include <lm/math.h>
#include <lm//surface.h>

LM_NAMESPACE_BEGIN(LM_NAMESPACE)

/*
    Rectangular portal

    p4 - p3
    |    |
    p1 - p2
*/
struct Portal {
    Vec3 p1, p2, p3, p4;    // Four points of the portal edges (counter-clockwised order)
    Vec3 e1;
    Vec3 e2;
    Vec3 n;                 // Portal orientation
    Float invA;             // Inverse area of the portal

    Portal() = default;
    Portal(Vec3 p1, Vec3 p2, Vec3 p4)
        : p1(p1) , p2(p2), p4(p4)
    {
        e1 = p2 - p1;
        e2 = p4 - p1;
        p3 = p1 + e1 + e2;
        const auto cr = glm::cross(e1, e2);
        n = glm::normalize(cr);
        const auto A = math::safe_sqrt(glm::length2(cr));
        invA = 1_f / A;
    }

    // Intersection with portal
    struct Hit {
        Float t;
        Vec2 uv;
    };
    std::optional<Hit> intersect(Ray r, Float tmin, Float tmax) const {
        auto p = glm::cross(r.d, e2);
        auto tv = r.o - p1;
        auto q = glm::cross(tv, e1);
        auto d = glm::dot(e1, p);
        auto ad = glm::abs(d);
        auto s = std::copysign(1_f, d);
        auto u = glm::dot(tv, p) * s;
        auto v = glm::dot(r.d, q) * s;
        if (ad < 1e-8_f || u < 0_f || v < 0_f || u > ad || v > ad) {
            return {};
        }
        auto t = glm::dot(e2, q) / d;
        if (t < tmin || tmax < t) {
            return {};
        }
        return Hit{ t, Vec2(u/ad, v/ad) };
    }

    // Sample a position on the portal
    Vec3 sample_position(Rng& rng) const {
        // Uniform sampling on the portal
        const auto p_portal = p1 + e1 * rng.u() + e2 * rng.u();
        return p_portal;
    }

    // Find a point on the portal on the segment.
    // If the segment does not intersect with the portal, returns nullopt.
    // v2.geom can be infinite
    std::optional<PointGeometry> intersect_with_segment(const PointGeometry& v1, const PointGeometry& v2) const {
        assert(!v1.infinite);
        Vec3 wo_y12;
        Float dist_y12;
        if (v2.infinite) {
            wo_y12 = -v2.wo;
            dist_y12 = Inf;
        }
        else {
            wo_y12 = glm::normalize(v2.p - v1.p);
            dist_y12 = glm::distance(v1.p, v2.p);
        }
        const auto hit = intersect({ v1.p, wo_y12 }, 0_f, dist_y12);
        if (!hit) {
            return {};
        }
        const auto p = v1.p + wo_y12 * hit->t;
        return PointGeometry::make_on_surface(p, n);
    }
};

LM_NAMESPACE_END(LM_NAMESPACE)