/*
    Lightmetrica - Copyright (c) 2019 Hisanari Otsu
    Distributed under MIT license. See LICENSE file for details.
*/

#include <lm/core.h>
#include <lm/renderer.h>
#include <lm/scene.h>
#include <lm/film.h>
#include <lm/scheduler.h>
#include <lm/parallel.h>
#include "mut.h"
#include "debug.h"

// Record acceptance ratio (overall and per-strategy)
#define MLT_STAT_ACCEPTANCE_RATIO 1
// Poll mutated paths
#define MLT_POLL_PATHS 0
// Record traceplot on the image plane
#define MLT_RECORD_TRACEPLOT_IMAGE_PLANE 1

LM_NAMESPACE_BEGIN(LM_NAMESPACE)

// A set of mutation strategies
class MutSet final : public Component {
private:
    std::vector<Ptr<Mut>> muts_;            // Underlying mutation strategies
    std::vector<Float> selection_weights_;  // Selection weights

public:
    virtual void construct(const Json& prop) override {
        // Load strategies and create selection dist
        for (const auto& [mut_name, weight] : prop["mut_weights"].items()) {
            muts_.push_back(comp::create<Mut>("mut::" + mut_name, make_loc(mut_name), prop));
            selection_weights_.push_back(weight);
        }
    }

private:
    Dist create_selection_dist(const Path& path) const {
        Dist dist;
        dist.clear();
        for (size_t i = 0; i < muts_.size(); i++) {
            const auto& mut = muts_[i];
            const auto w = selection_weights_[i];
            dist.add(mut->check_mutatable(path) ? w : 0_f);
        }
        if (dist.c.back() == 0_f) {
            LM_THROW_EXCEPTION(Error::FailedToRender, "No mutation strategy can mutate the path.");
        }
        dist.norm();
        return dist;
    }

public:
    // Number of mutation strategies
    int num_strategies() const {
        return (int)(muts_.size());
    }

    // Get strategy by index
    const Mut* get_strategy_by_index(int i) const {
        return muts_[i].get();
    }

    // Select a mutation strategy
    struct SelectedMut {
        const Mut* p;
        int index;
    };
    SelectedMut select_mut(Rng& rng, const Path& path) {
        const auto selection_dist = create_selection_dist(path);
        const int i = selection_dist.sample(rng.u());
        return { muts_[i].get(), i };
    }

    Float selection_prob(int index, const Path& path) {
        const auto selection_dist = create_selection_dist(path);
        return selection_dist.pmf(index);
    }
};

LM_COMP_REG_IMPL(MutSet, "mutset::default");

// ------------------------------------------------------------------------------------------------

// Metropolis light transport (MLT)
class Renderer_MLT final : public Renderer {
private:
    Scene* scene_;                      // Reference to scene asset
    Film* film_;                        // Reference to film asset for output
    int min_verts_;                     // Minimum number of path vertices
    int max_verts_;                     // Maximum number of path vertices
    std::optional<unsigned int> seed_;  // Random seed
    Ptr<scheduler::Scheduler> sched_;   // Scheduler for parallel processing
    Float normalization_;               // Normalization factor
    Ptr<MutSet> mutset_;                // Mutation strategies

    #if MLT_RECORD_TRACEPLOT_IMAGE_PLANE
    long long num_traceplot_image_plane_records_;
    long long num_traceplot_record_per_mutation_;
    #endif

public:
    virtual void construct(const Json& prop) override {
        scene_ = json::comp_ref<Scene>(prop, "scene");
        film_ = json::comp_ref<Film>(prop, "output");
        scene_->camera()->set_aspect_ratio(film_->aspect());
        min_verts_ = json::value<int>(prop, "min_verts");
        max_verts_ = json::value<int>(prop, "max_verts");
        seed_ = json::value_or_none<unsigned int>(prop, "seed");
        const auto sched_name = json::value<std::string>(prop, "scheduler");
        sched_ = comp::create<scheduler::Scheduler>(
            "scheduler::spi::" + sched_name, make_loc("scheduler"), prop);
        normalization_ = json::value<Float>(prop, "normalization");
        mutset_ = comp::create<MutSet>("mutset::default", make_loc("mutset"), prop);
        #if MLT_RECORD_TRACEPLOT_IMAGE_PLANE
        num_traceplot_image_plane_records_ = json::value<long long>(prop, "num_traceplot_image_plane_records");
        num_traceplot_record_per_mutation_ = json::value<long long>(prop, "num_traceplot_record_per_mutation");
        #endif
    }

private:
    // Generate initial state with BDPT
    Path generate_init_path(Rng& rng) const {
        while (true) {
            const auto subpathE = path::sample_subpath(rng, scene_, max_verts_, TransDir::EL);
            const auto subpathL = path::sample_subpath(rng, scene_, max_verts_, TransDir::LE);
            const int nE = subpathE.num_verts();
            const int nL = subpathL.num_verts();
            for (int s = 0; s <= nL; s++) {
                for (int t = 0; t <= nE; t++) {
                    const int k = s + t;
                    if (k < min_verts_ || max_verts_ < k) {
                        continue;
                    }
                    const auto path = path::connect_subpaths(scene_, subpathL, subpathE, s, t);
                    if (!path) {
                        continue;
                    }
                    const auto f = path->eval_measurement_contrb_bidir(scene_, s);
                    if (math::is_zero(f)) {
                        continue;
                    }
                    const auto p = path->pdf_bidir(scene_, s);
                    if (p == 0_f) {
                        continue;
                    }
                    return *path;
                }
            }
        }
    }

public:
    virtual Json render() const override {
        scene_->require_renderable();
        film_->clear();
        const auto size = film_->size();

        // Random number generator for initialization
        const auto seed = seed_ ? *seed_ : math::rng_seed();
        LM_INFO("Seed: {}", seed);
        Rng init_rng(seed);

        // ----------------------------------------------------------------------------------------

        // TLS
        struct Context {
            Rng rng;
            Path curr;
        };
        std::vector<Context> contexts(parallel::num_threads());
        for (int i = 0; i < (int)(contexts.size()); i++) {
            auto& ctx = contexts[i];
            ctx.rng = Rng(init_rng.u_int());
            ctx.curr = generate_init_path(init_rng);
        }

        // ----------------------------------------------------------------------------------------

        #if MLT_STAT_ACCEPTANCE_RATIO
        std::vector<long long> sample_counts(mutset_->num_strategies(), 0);
        std::vector<long long> accept_counts(mutset_->num_strategies(), 0);
        #endif

        #if MLT_RECORD_TRACEPLOT_IMAGE_PLANE
        std::vector<Vec2> traceplot_image_plane;
        long long traceplot_record_count = 0;
        #endif

        // ----------------------------------------------------------------------------------------

        // Execute parallel process
        const auto processed = sched_->run([&](long long, long long sample_index, int thread_id) {
            auto& ctx = contexts[thread_id];
            LM_UNUSED(sample_index);

            #if MLT_POLL_PATHS
            if (thread_id == 0) {
                debug::poll({
                    {"id", "path"},
                    {"sample_index", sample_index},
                    {"path", ctx.curr}
                });
            }
            #endif

            #if MLT_RECORD_TRACEPLOT_IMAGE_PLANE
            if (thread_id == 0 &&
                (long long)traceplot_image_plane.size() < num_traceplot_image_plane_records_ &&
                (long long)--traceplot_record_count <= 0) {
                const auto rp = ctx.curr.raster_position(scene_);
                traceplot_image_plane.push_back(rp);
                traceplot_record_count = num_traceplot_record_per_mutation_;
            }
            #endif

            const auto [accept, strategy_index] = [&]() -> std::tuple<bool, int> {
                // Select a mutation strategy
                const auto [mut, strategy_index] = mutset_->select_mut(ctx.rng, ctx.curr);

                // Sample proposal
                const auto prop = mut->sample_proposal(ctx.rng, ctx.curr);
                if (!prop) {
                    return { false, strategy_index };
                }

                // MH update
                const auto Qxy =
                    mut->eval_Q(ctx.curr, prop->path, prop->subspace) *
                    mutset_->selection_prob(strategy_index, ctx.curr);
                const auto Qyx =
                    mut->eval_Q(prop->path, ctx.curr, mut->reverse_subspace(prop->subspace)) *
                    mutset_->selection_prob(strategy_index, prop->path);
                const auto A = (Qxy == 0_f || Qyx == 0_f) ? 0_f : std::min(1_f, Qyx / Qxy);
                if (ctx.rng.u() < A) {
                    // Accepted
                    ctx.curr = prop->path;
                    return { true, strategy_index };
                }
                else {
                    // Rejected
                    #if MLT_POLL_PATHS
                    // Record proposed but rejeced paths
                    if (thread_id == 0) {
                        debug::poll({
                            {"id", "rejected_path"},
                            {"sample_index", sample_index},
                            {"path", prop->path}
                        });
                    }
                    #endif
                    return { false, strategy_index };
                }
            }();

            #if MLT_STAT_ACCEPTANCE_RATIO
            // Record statistics
            if (thread_id == 0) {
                sample_counts[strategy_index]++;
                if (accept) {
                    accept_counts[strategy_index]++;
                }
            }
            #endif

            // Accumulate contribution
            const auto contrb = ctx.curr.eval_measurement_contrb_bidir(scene_, 0);
            if (!math::is_zero(contrb)) {
                const auto rp = ctx.curr.raster_position(scene_);
                const auto C = contrb * (normalization_ / path::scalar_contrb(contrb));
                film_->splat(rp, C);
            }
        });
        
        // ----------------------------------------------------------------------------------------

        // Rescale film
        film_->rescale(Float(size.w*size.h) / processed);

        #if MLT_STAT_ACCEPTANCE_RATIO
        Json acceptance_ratio;
        long long accept_count_total = 0;
        long long sample_count_total = 0;
        for (int i = 0; i < mutset_->num_strategies(); i++) {
            accept_count_total += accept_counts[i];
            sample_count_total += sample_counts[i];
            Float acc_ratio = -1_f;
            if (sample_count_total > 0) {
                acc_ratio = (Float)(accept_counts[i]) / sample_counts[i];
            }
            const auto* mut = mutset_->get_strategy_by_index(i);
            acceptance_ratio[mut->key()] = acc_ratio;
        }
        #endif

        return {
            {"processed", processed},
            #if MLT_STAT_ACCEPTANCE_RATIO
            {"overall_acceptance_ratio", (Float)(accept_count_total) / sample_count_total},
            {"acceptance_ratio", acceptance_ratio},
            #endif
            #if MLT_RECORD_TRACEPLOT_IMAGE_PLANE
            {"traceplot_image_plane", traceplot_image_plane}
            #endif
        };
    }
};

LM_COMP_REG_IMPL(Renderer_MLT, "renderer::mlt");

LM_NAMESPACE_END(LM_NAMESPACE)
