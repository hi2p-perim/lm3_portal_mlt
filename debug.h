/*
    Lightmetrica - Copyright (c) 2019 Hisanari Otsu
    Distributed under MIT license. See LICENSE file for details.
*/

#include <lm/debug.h>
#include <lm/math.h>

LM_NAMESPACE_BEGIN(nlohmann)
// Specialize adl_serializer for one-way automatic conversion from path to Json type
template <>
struct adl_serializer<lm::Path> {
    static void to_json(lm::Json& j, const lm::Path& path) {
        for (size_t i = 0; i < path.vs.size(); i++) {
            const auto& v = path.vs[i];
            if (i == 0 && v.sp.geom.infinite) {
                const auto& vn = path.vs[i + 1];
                const auto p = vn.sp.geom.p - 100.0 * v.sp.geom.wo;
                j.push_back(p);
            }
            else {
                j.push_back(v.sp.geom.p);
            }
        }
    }
};
LM_NAMESPACE_END(nlohmann)
