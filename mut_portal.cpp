/*
    Lightmetrica - Copyright (c) 2019 Hisanari Otsu
    Distributed under MIT license. See LICENSE file for details.
*/

#include <lm/core.h>
#include <lm/parallel.h>
#include <lm/mesh.h>
#include "mut.h"
#include "portal.h"
#include "debug.h"

#define MUT_PORTAL_POLL_PROPOSED_PATH 0
#define MUT_PORTAL_PETURBED_DIR 0

LM_NAMESPACE_BEGIN(LM_NAMESPACE)

// Portal-based path perturbation
class Mut_Portal final : public Mut {
private:
    Scene* scene_;
    Float s1_;
    Float s2_;
    Portal portal_;

private:
    // Check intersection between edge and portal
    // i: edge index from camera
    std::optional<PointGeometry> intersect_portal(const Path& path, int i) const {
        const auto* v1 = path.vertex_at(i, TransDir::EL);
        const auto* v2 = path.vertex_at(i+1, TransDir::EL);
        assert(!(v1->sp.geom.infinite && v2->sp.geom.infinite));
        return portal_.intersect_with_segment(v1->sp.geom, v2->sp.geom);
    }

    // Find the first edge with portal from camera
    struct PortalEdge {
        int i;
        PointGeometry geom;
    };
    std::optional<PortalEdge> find_first_portal_edge(const Path& path) const {
        // Skip the primary edge because if the camera is almost always degenerated.
        // This can avoid a tricky case which
        // the proposed path cannot be mutated back to the original state
        // due to the missing intersection with camera.
        for (int i = 1; i < path.num_edges(); i++) {
            const auto hit = intersect_portal(path, i);
            if (hit) {
                return PortalEdge{ i, *hit };
            }
        }
        return {};
    }

public:
    virtual void construct(const Json& prop) override {
        scene_ = json::comp_ref<Scene>(prop, "scene");
        s1_ = json::value<Float>(prop, "s1");
        s2_ = json::value<Float>(prop, "s2");
        auto ps = prop["portal"];
        if (ps.is_array()) {
            // The portal is specified by array of vec3
            portal_ = Portal(ps[0], ps[1], ps[2]);
        }
        else if (ps.is_string()) {
            // The portal is specified by mesh
            // Extract 4 vertices from the mesh
            auto* portal_mesh = json::comp_ref<Mesh>(prop, "portal");
            if (portal_mesh->num_triangles() != 2) {
                LM_THROW_EXCEPTION(Error::InvalidArgument, "Portal is not quad");
            }
            const auto tri = portal_mesh->triangle_at(0);

            /*
                   p3
                 / |
              p1 - p2
            */
            portal_ = Portal(tri.p2.p, tri.p3.p, tri.p1.p);
        }
        else {
            LM_THROW_EXCEPTION(Error::InvalidArgument, "Invalid type for portal parameter");
        }
    }

    virtual bool check_mutatable(const Path& curr) const override {
        // Find the first edge intersected with portal from the camera
        const auto pe = find_first_portal_edge(curr);
        if (!pe) {
            // Not mutatable if the portal is not intersected with the path,
            // or intersected with the primary edge from the camera.
            return false;
        }

        // Not mutatable if connected vertices are specular
        const auto* vE  = curr.vertex_at(pe->i,   TransDir::EL);
        const auto* vEp = curr.vertex_at(pe->i-1, TransDir::EL);
        const auto* vL  = curr.vertex_at(pe->i+1, TransDir::EL);
        const auto* vLp = curr.vertex_at(pe->i+2, TransDir::EL);
        if (vE->is_specular(scene_) || vL->is_specular(scene_)) {
            return false;
        }
        if (vEp && vEp->is_specular(scene_)) {
            return false;
        }
        if (vLp && vLp->is_specular(scene_)) {
            return false;
        }

        return true;
    }

    virtual Subspace reverse_subspace(const Subspace& subspace) const override {
        return subspace;
    }

    virtual std::optional<Proposal> sample_proposal(Rng& rng, const Path& curr) const override {
        if (!check_mutatable(curr)) {
            return {};
        }

        // Find the first edge with portal from the camera
        const auto pe = find_first_portal_edge(curr);
        assert(pe);

        // Vertices around an edge with portal
        // ... - vE_prev - vE - (portal) - vL - vL_prev ...
        const auto* xE = curr.vertex_at(pe->i, TransDir::EL);
        const auto* xL = curr.vertex_at(pe->i+1, TransDir::EL);
        const auto* xE_prev = curr.vertex_at(pe->i-1, TransDir::EL);
        const auto* xL_prev = curr.vertex_at(pe->i+2, TransDir::EL);
        assert(xE);
        assert(xL);
        
        // Perturb a direction from the portal
        const auto prop_wo_portal = [&]() -> Vec3 {
            const auto curr_wo = curr.direction(xE, xL);
            const auto prop_wo = path::perturb_direction_truncated_reciprocal(rng, curr_wo, s1_, s2_);
            return prop_wo;
        }();

        #if MUT_PORTAL_PETURBED_DIR
        if (parallel::main_thread()) {
            debug::poll({
                {"id", "mut_portal_perturbed_ray"},
                {"o", pe->geom.p},
                {"d", prop_wo_portal}});
        }
        #endif
        
        // Trace rays in two directions
        const auto hit_yL = scene_->intersect({ pe->geom.p, prop_wo_portal });
        if (!hit_yL) {
            return {};
        }
        const auto hit_yE = scene_->intersect({ pe->geom.p, -prop_wo_portal });
        if (!hit_yE) {
            return {};
        }
        if (hit_yL->geom.infinite && hit_yE->geom.infinite) {
            return {};
        }

        // Check if connectable
        // Component index are fixed by the current vertices
        if (path::is_specular_component(scene_, *hit_yL, xL->comp)) {
            return {};
        }
        if (path::is_specular_component(scene_, *hit_yE, xE->comp)) {
            return {};
        }

        // Handle a corner case where the ray
        // hits with environment, but there's already previous vertex.
        if (hit_yL->geom.infinite && xL_prev) {
            return {};
        }
        if (hit_yE->geom.infinite && xE_prev) {
            return {};
        }
 
        // Check visibility
        if (xE_prev && !scene_->visible(xE_prev->sp, *hit_yE)) {
            return {};
        }
        if (xL_prev && !scene_->visible(xL_prev->sp, *hit_yL)) {
            return {};
        }

        // Construct the proposed path
        Path prop;
        for (int i = curr.num_verts()-1; i >= pe->i+2; i--) {
            prop.vs.push_back(*curr.vertex_at(i, TransDir::EL));
        }
        prop.vs.push_back({ *hit_yL, xL->comp });
        prop.vs.push_back({ *hit_yE, xE->comp });
        for (int i = pe->i-1; i>=0; i--) {
            prop.vs.push_back(*curr.vertex_at(i, TransDir::EL));
        }
        {
            auto& vL = prop.vs.front();
            if (!scene_->is_light(vL.sp)) {
                return {};
            }
            auto& vE = prop.vs.back();
            if (!scene_->is_camera(vE.sp)) {
                return {};
            }
            vL.sp = vL.sp.as_type(SceneInteraction::LightEndpoint);
            vE.sp = vE.sp.as_type(SceneInteraction::CameraEndpoint);
        }

        #if MUT_PORTAL_POLL_PROPOSED_PATH
        if (parallel::main_thread()) {
            debug::poll({
                {"id", "proposed_path"},
                {"path", prop}});
        }
        #endif

        assert(curr.num_verts() == prop.num_verts());
        return Proposal{ prop, {} };
    }

    virtual Float eval_Q(const Path&, const Path& y, const Subspace&) const override {
        // Find the first edge with portal from the camera
        const auto pe = find_first_portal_edge(y);
        if (!pe) {
            return 0_f;
        }

        // The most of the terms are cancelled out. See the derivation for detail.
        const auto f = y.eval_measurement_contrb_bidir(scene_, pe->i+1);
        if (math::is_zero(f)) {
            return 0_f;
        }
        
        // Geometry term
        const auto* yE = y.vertex_at(pe->i, TransDir::EL);
        const auto* yL = y.vertex_at(pe->i+1, TransDir::EL);
        const auto G = surface::geometry_term(yE->sp.geom, yL->sp.geom);

        const auto C = f / G;
        return 1_f / path::scalar_contrb(C);
    }
};

LM_COMP_REG_IMPL(Mut_Portal, "mut::portal");

LM_NAMESPACE_END(LM_NAMESPACE)
