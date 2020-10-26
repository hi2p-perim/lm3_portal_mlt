/*
    Lightmetrica - Copyright (c) 2019 Hisanari Otsu
    Distributed under MIT license. See LICENSE file for details.
*/

#include <lm/core.h>
#include "mut.h"

// Poll failed connections
#define MUT_LENS_POLL_FAILED_CONNECTION 1

#if MUT_LENS_POLL_FAILED_CONNECTION
#include <lm/debug.h>
#include <lm/parallel.h>
#endif

LM_NAMESPACE_BEGIN(LM_NAMESPACE)

// Lens perturbation
class Mut_Lens final : public Mut {
private:
    Scene* scene_;
    Float s1_;      // Lower bound of the mutation range
    Float s2_;      // Upper bound of the mutation range

public:
    virtual void construct(const Json& prop) override {
        scene_ = json::comp_ref<Scene>(prop, "scene");
        s1_ = json::value<Float>(prop, "s1");
        s2_ = json::value<Float>(prop, "s2");
    }

private:
    // Find the index of first non-S vertex from camera
    int find_first_non_s(const Path& path) const {
        const int n = path.num_verts();
        int i = 1;
        while (i < n && path.vertex_at(i, TransDir::EL)->is_specular(scene_)) {
            i++;
        }
        return i;
    }

public:
    virtual bool check_mutatable(const Path& curr) const override {
        const int n = curr.num_verts();

        // Find first non-S vertex
        const int i = find_first_non_s(curr);
        
        // Path is not mutable if non-S vertex is found on midpoints
        // and the next vertex is S.
        if (i+1 < n && curr.vertex_at(i + 1, TransDir::EL)->is_specular(scene_)) {
            return false;
        }
        
        // Oherwise the path is mutatable.
        // Example of the mutatable paths: ESSDL, ESSL.
        return true;
    }

    virtual std::optional<Proposal> sample_proposal(Rng& rng, const Path& curr) const override {
        // Number of vertices in the current path
        const int curr_n = curr.num_verts();

        // Check if the path is mutatble with this strategy
        if (!check_mutatable(curr)) {
            return {};
        }

        // Find first non-S vertex from camera
        const int first_non_s_ind = find_first_non_s(curr);
        
        // Perturb eye subpath
        const auto subpathE = [&]() -> std::optional<Path> {
            Path subpathE;

            // Keep initial vertex
            subpathE.vs.push_back(*curr.vertex_at(0, TransDir::EL));

            // Trace rays until it hit with non-S surface 
            // with the same number of non-S vertices as the current path.
            for (int i = 1; i <= first_non_s_ind; i++) {
                // Vertices
                const auto* v = subpathE.subpath_vertex_at(i-1);
                const auto* v_prev = subpathE.subpath_vertex_at(i-2);
                const auto wi = subpathE.direction(v, v_prev);

                // Perturb or sample the next direction
                Vec3 wo;
                if (i == 1) {
                    // Perturb primary ray direction
                    wo = [&] {
                        const auto* curr_v0 = curr.vertex_at(0, TransDir::EL);
                        const auto* curr_v1 = curr.vertex_at(1, TransDir::EL);
                        const auto curr_wo = curr.direction(curr_v0, curr_v1);
                        const auto prop_wo = path::perturb_direction_truncated_reciprocal(rng, curr_wo, s1_, s2_);
                        return prop_wo;
                    }();
                }
                else {
                    // Sample next direction on specular vertex
                    // This is essentially deterministic operation
                    const auto s = path::sample_direction(rng.next<path::DirectionSampleU>(), scene_, v->sp, wi, v->comp, TransDir::EL);
                    if (!s || math::is_zero(s->weight)) {
                        return {};
                    }
                    wo = s->wo;
                }

                // Intersection to the next surface
                const auto hit = scene_->intersect({ v->sp.geom.p, wo });
                if (!hit) {
                    // Rejected due to the change of path length
                    return {};
                }

                // Use component index same as the current path
                const auto* curr_v_next = curr.vertex_at(i, TransDir::EL);
                const int comp = curr_v_next->comp;

                // Terminate if it finds non-S vertex at i < first_non_s_ind
                const bool is_specular = path::is_specular_component(scene_, *hit, comp);
                if (i < first_non_s_ind && !is_specular) {
                    return {};
                }

                // Terminate if it finds S vertex at i = first_non_s_ind
                if (i == first_non_s_ind && is_specular) {
                    return {};
                }

                // Add a vertex
                subpathE.vs.push_back({ *hit, comp });
            }

            return subpathE;
        }();
        if (!subpathE) {
            return {};
        }

        // Number of vertices in each subpath
        const int nE = subpathE->num_verts();
        const int nL = curr_n - nE;
        assert(nE == first_non_s_ind + 1);

        // Light subpath
        Path subpathL;
        for (int i = 0; i < nL; i++) {
            subpathL.vs.push_back(*curr.vertex_at(i, TransDir::LE));
        }

        // Generate proposal
        const auto prop_path = path::connect_subpaths(scene_, subpathL, *subpathE, nL, nE);
        if (!prop_path) {
            #if MUT_LENS_POLL_FAILED_CONNECTION
            if (nL > 0 && nE > 0 && parallel::main_thread()) {
                debug::poll({
                    {"id", "mut_lens_failed_connection"},
                    {"v1", subpathL.vs[nL-1].sp.geom.p},
                    {"v2", subpathE->vs[nE-1].sp.geom.p}
                });
            }
            #endif
            return {};
        }

        return Proposal{*prop_path, {}};
    }

    virtual Subspace reverse_subspace(const Subspace& subspace) const override {
        return subspace;
    }
    
    virtual Float eval_Q(const Path&, const Path& y, const Subspace&) const override {
        // Number of vertices in each path
        const int x_n = y.num_verts();
        const int y_n = y.num_verts();
        assert(x_n == y_n);
        const int n = x_n;

        // Find first non-S vertex from camera
        const int first_non_s_ind = find_first_non_s(y);
        const int nE = first_non_s_ind + 1;
        const int nL = n - nE;

        // Evaluate terms
        const auto f = y.eval_measurement_contrb_bidir(scene_, nL);
        if (math::is_zero(f)) {
            return 0_f;
        }

        // Transition probability
        // Trasition kernel for primary ray will be cancelled out.
        auto p = 1_f;
        for (int i = 0; i < nE - 1; i++) {
            const auto* v      = y.vertex_at(i,   TransDir::EL);
            const auto* v_prev = y.vertex_at(i-1, TransDir::EL);
            const auto* v_next = y.vertex_at(i+1, TransDir::EL);
            const auto wi = y.direction(v, v_prev);
            const auto wo = y.direction(v, v_next);
            const auto p_projSA = i == 0 ? 1_f :  path::pdf_direction(scene_, v->sp, wi, wo, v->comp, false);
            p *= surface::convert_pdf_to_area(p_projSA, v->sp.geom, v_next->sp.geom);
        }
        const auto C = f / p;
        return 1_f / path::scalar_contrb(C);
    }
};

LM_COMP_REG_IMPL(Mut_Lens, "mut::lens");

LM_NAMESPACE_END(LM_NAMESPACE)
