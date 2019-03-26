// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/global_pooling_2d_layer.hpp"

#include <algorithm>
#include <limits>
#include <string>

namespace fdeep { namespace internal
{

class global_max_pooling_2d_layer : public global_pooling_2d_layer
{
public:
    explicit global_max_pooling_2d_layer(const std::string& name) :
    global_pooling_2d_layer(name)
    {
    }
protected:
    tensor5 pool(const tensor5& in) const override
    {
        tensor5 out(shape5(1, 1, 1, 1, in.shape().depth_), 0);
        for (std::size_t z = 0; z < out.shape().depth_; ++z)
        {
            float_type val = std::numeric_limits<float_type>::lowest();
            for (std::size_t y = 0; y < in.shape().height_; ++y)
            {
                for (std::size_t x = 0; x < in.shape().width_; ++x)
                {
                    val = std::max(val, in.get(0, 0, y, x, z));
                }
            }
            out.set(0, 0, 0, 0, z, val);
        }
        return out;
    }
};

} } // namespace fdeep, namespace internal
