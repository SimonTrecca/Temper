/**
 * @file Stats.cpp
 * @brief Statistical distributions utilities definitions.
 */

#include "temper/Stats.hpp"
#include "temper/Utils.hpp"
#include "temper/SYCLUtils.hpp"
#include "temper/Math.hpp"

#include <random>

namespace temper::stats
{

template<typename value_t>
Tensor<value_t> randn(const std::vector<uint64_t>& out_shape,
    MemoryLocation res_loc,
    uint64_t seed)
{
    Tensor<value_t> loc({1}, res_loc);
    loc = std::vector<value_t>{static_cast<value_t>(0.0)};

    Tensor<value_t> scale({1}, res_loc);
    scale = std::vector<value_t>{static_cast<value_t>(1.0)};

    return norm::rvs<value_t>(loc, scale, out_shape, res_loc, seed);
}
template Tensor<float> randn<float>
(const std::vector<uint64_t>&, MemoryLocation, uint64_t);

namespace norm
{

template<typename value_t>
Tensor<value_t> pdf(const Tensor<value_t>& x,
    const Tensor<value_t>& loc,
    const Tensor<value_t>& scale)
{
    const std::vector<uint64_t> & x_shape = x.get_dimensions();
    const std::vector<uint64_t> & loc_shape = loc.get_dimensions();
    const std::vector<uint64_t> & scale_shape = scale.get_dimensions();

    if (x_shape.empty())
    {
        throw std::invalid_argument(R"(norm::pdf:
            x tensor has no elements.)");
    }
    if (loc_shape.empty())
    {
        throw std::invalid_argument(R"(norm::pdf:
            loc tensor has no elements.)");
    }
    if (scale_shape.empty())
    {
        throw std::invalid_argument(R"(norm::pdf:
            scale tensor has no elements.)");
    }

    temper::utils::TensorDesc a_desc{x_shape, x.get_strides()};
    temper::utils::TensorDesc b_desc{loc_shape, loc.get_strides()};
    temper::utils::TensorDesc c_desc{scale_shape, scale.get_strides()};

    temper::utils::BroadcastResult res =
        temper::utils::compute_broadcast({a_desc, b_desc, c_desc});

    const std::vector<uint64_t> out_shape = std::move(res.shape);
    const int64_t out_rank = static_cast<int64_t>(out_shape.size());

    const std::vector<uint64_t> x_bcast = std::move(res.strides[0]);
    const std::vector<uint64_t> loc_bcast = std::move(res.strides[1]);
    const std::vector<uint64_t> scale_bcast = std::move(res.strides[2]);

    Tensor<value_t> result(out_shape, x.get_memory_location());

    uint64_t* p_out_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));
    uint64_t* p_x_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));
    uint64_t* p_loc_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));
    uint64_t* p_scale_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));

    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    bool alloc_ok = (p_out_divs && p_x_bcast && p_loc_bcast &&
                     p_scale_bcast && p_error_flag);

    if (!alloc_ok)
    {
        sycl::free(p_out_divs, g_sycl_queue);
        sycl::free(p_x_bcast, g_sycl_queue);
        sycl::free(p_loc_bcast, g_sycl_queue);
        sycl::free(p_scale_bcast, g_sycl_queue);
        sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    const std::vector<uint64_t> out_divs = std::move(res.divisors);

    g_sycl_queue.memcpy(p_out_divs,
        out_divs.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();
    g_sycl_queue.memcpy(p_x_bcast,
        x_bcast.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();
    g_sycl_queue.memcpy(p_loc_bcast,
        loc_bcast.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();
    g_sycl_queue.memcpy(p_scale_bcast,
        scale_bcast.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();

    *p_error_flag = 0;

    const value_t* p_x = x.get_data();
    const value_t* p_loc = loc.get_data();
    const value_t* p_scale = scale.get_data();
    value_t* p_out = result.get_data();

    const double inv_sqrt_2pi = 1.0 / std::sqrt(2.0 * M_PI);

    const uint64_t total_output_elems = result.get_num_elements();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);

            const uint64_t x_idx = temper::sycl_utils::idx_of(
                flat, p_out_divs, p_x_bcast, out_rank);
            const uint64_t loc_idx = temper::sycl_utils::idx_of(
                flat, p_out_divs, p_loc_bcast, out_rank);
            const uint64_t scale_idx = temper::sycl_utils::idx_of(
                flat, p_out_divs, p_scale_bcast, out_rank);

            double xp = static_cast<double>(p_x[x_idx]);
            double locp = static_cast<double>(p_loc[loc_idx]);
            double scalep = static_cast<double>(p_scale[scale_idx]);

            temper::sycl_utils::device_check_nan_and_set<double>
                (xp, p_error_flag);
            temper::sycl_utils::device_check_nan_and_set<double>
                (locp, p_error_flag);
            temper::sycl_utils::device_check_nan_and_set<double>
                (scalep, p_error_flag);

            if (scalep <= 0.0)
            {
                p_error_flag[0] = 4;
                p_out[flat] = std::numeric_limits<value_t>::quiet_NaN();
                return;
            }

            double z = (xp - locp) / scalep;
            double ex = sycl::exp(-0.5 * z * z);
            double outv = (inv_sqrt_2pi / scalep) * ex;

            temper::sycl_utils::device_check_finite_and_set<double>
                (outv, p_error_flag);
            p_out[flat] = static_cast<value_t>(outv);
        });
    }).wait();

    int32_t err = *p_error_flag;

    sycl::free(p_out_divs, g_sycl_queue);
    sycl::free(p_x_bcast, g_sycl_queue);
    sycl::free(p_loc_bcast, g_sycl_queue);
    sycl::free(p_scale_bcast, g_sycl_queue);
    sycl::free(p_error_flag, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::invalid_argument(R"(norm::pdf: NaN detected in inputs.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(norm::pdf:
                non-finite result (overflow or Inf) produced.)");
        }
        if (err == 4)
        {
            throw std::invalid_argument(R"(norm::pdf: scale must be positive.)");
        }
        throw std::runtime_error(R"(norm::pdf:
            numeric error during pdf computation.)");
    }

    return result;
}
template Tensor<float> pdf<float>
(const Tensor<float>&, const Tensor<float>&, const Tensor<float>&);

template<typename value_t>
Tensor<value_t> logpdf(const Tensor<value_t>& x,
    const Tensor<value_t>& loc,
    const Tensor<value_t>& scale)
{
    const std::vector<uint64_t> & x_shape = x.get_dimensions();
    const std::vector<uint64_t> & loc_shape = loc.get_dimensions();
    const std::vector<uint64_t> & scale_shape = scale.get_dimensions();

    if (x_shape.empty())
    {
        throw std::invalid_argument(R"(norm::logpdf:
            x tensor has no elements.)");
    }
    if (loc_shape.empty())
    {
        throw std::invalid_argument(R"(norm::logpdf:
            loc tensor has no elements.)");
    }
    if (scale_shape.empty())
    {
        throw std::invalid_argument(R"(norm::logpdf:
            scale tensor has no elements.)");
    }

    temper::utils::TensorDesc a_desc{x_shape, x.get_strides()};
    temper::utils::TensorDesc b_desc{loc_shape, loc.get_strides()};
    temper::utils::TensorDesc c_desc{scale_shape, scale.get_strides()};

    temper::utils::BroadcastResult res =
        temper::utils::compute_broadcast({a_desc, b_desc, c_desc});

    const std::vector<uint64_t> out_shape = std::move(res.shape);
    const int64_t out_rank = static_cast<int64_t>(out_shape.size());

    const std::vector<uint64_t> x_bcast = std::move(res.strides[0]);
    const std::vector<uint64_t> loc_bcast = std::move(res.strides[1]);
    const std::vector<uint64_t> scale_bcast = std::move(res.strides[2]);

    Tensor<value_t> result(out_shape, x.get_memory_location());

    uint64_t* p_out_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));
    uint64_t* p_x_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));
    uint64_t* p_loc_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));
    uint64_t* p_scale_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));

    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    bool alloc_ok = (p_out_divs && p_x_bcast && p_loc_bcast &&
                     p_scale_bcast && p_error_flag);

    if (!alloc_ok)
    {
        sycl::free(p_out_divs, g_sycl_queue);
        sycl::free(p_x_bcast, g_sycl_queue);
        sycl::free(p_loc_bcast, g_sycl_queue);
        sycl::free(p_scale_bcast, g_sycl_queue);
        sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    const std::vector<uint64_t> out_divs = std::move(res.divisors);

    g_sycl_queue.memcpy(p_out_divs,
        out_divs.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();
    g_sycl_queue.memcpy(p_x_bcast,
        x_bcast.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();
    g_sycl_queue.memcpy(p_loc_bcast,
        loc_bcast.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();
    g_sycl_queue.memcpy(p_scale_bcast,
        scale_bcast.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();

    *p_error_flag = 0;

    const value_t* p_x = x.get_data();
    const value_t* p_loc = loc.get_data();
    const value_t* p_scale = scale.get_data();
    value_t* p_out = result.get_data();

    const double log_sqrt_2pi = 0.5 * std::log(2.0 * M_PI);

    const uint64_t total_output_elems = result.get_num_elements();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);

            const uint64_t x_idx = temper::sycl_utils::idx_of(
                flat, p_out_divs, p_x_bcast, out_rank);
            const uint64_t loc_idx = temper::sycl_utils::idx_of(
                flat, p_out_divs, p_loc_bcast, out_rank);
            const uint64_t scale_idx = temper::sycl_utils::idx_of(
                flat, p_out_divs, p_scale_bcast, out_rank);

            double xp = static_cast<double>(p_x[x_idx]);
            double locp = static_cast<double>(p_loc[loc_idx]);
            double scalep = static_cast<double>(p_scale[scale_idx]);

            temper::sycl_utils::device_check_nan_and_set<double>
                (xp, p_error_flag);
            temper::sycl_utils::device_check_nan_and_set<double>
                (locp, p_error_flag);
            temper::sycl_utils::device_check_nan_and_set<double>
                (scalep, p_error_flag);

            if (scalep <= 0.0)
            {
                p_error_flag[0] = 4;
                p_out[flat] = std::numeric_limits<value_t>::quiet_NaN();
                return;
            }

            double z = (xp - locp) / scalep;

            double outv = -0.5 * z * z - sycl::log(scalep) - log_sqrt_2pi;

            temper::sycl_utils::device_check_finite_and_set<double>
                (outv, p_error_flag);
            p_out[flat] = static_cast<value_t>(outv);
        });
    }).wait();

    int32_t err = *p_error_flag;

    sycl::free(p_out_divs, g_sycl_queue);
    sycl::free(p_x_bcast, g_sycl_queue);
    sycl::free(p_loc_bcast, g_sycl_queue);
    sycl::free(p_scale_bcast, g_sycl_queue);
    sycl::free(p_error_flag, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::invalid_argument(R"(norm::logpdf:
                NaN detected in inputs.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(norm::logpdf:
                non-finite result (overflow or Inf) produced.)");
        }
        if (err == 4)
        {
            throw std::invalid_argument(R"(norm::logpdf:
                scale must be positive.)");
        }
        throw std::runtime_error(R"(norm::logpdf:
            numeric error during logpdf computation.)");
    }

    return result;
}
template Tensor<float> logpdf<float>
(const Tensor<float>&, const Tensor<float>&, const Tensor<float>&);

template<typename value_t>
Tensor<value_t> cdf(const Tensor<value_t>& x,
    const Tensor<value_t>& loc,
    const Tensor<value_t>& scale)
{
    const std::vector<uint64_t> & x_shape = x.get_dimensions();
    const std::vector<uint64_t> & loc_shape = loc.get_dimensions();
    const std::vector<uint64_t> & scale_shape = scale.get_dimensions();

    if (x_shape.empty())
    {
        throw std::invalid_argument(R"(norm::cdf:
            x tensor has no elements.)");
    }
    if (loc_shape.empty())
    {
        throw std::invalid_argument(R"(norm::cdf:
            loc tensor has no elements.)");
    }
    if (scale_shape.empty())
    {
        throw std::invalid_argument(R"(norm::cdf:
            scale tensor has no elements.)");
    }

    temper::utils::TensorDesc a_desc{x_shape, x.get_strides()};
    temper::utils::TensorDesc b_desc{loc_shape, loc.get_strides()};
    temper::utils::TensorDesc c_desc{scale_shape, scale.get_strides()};

    temper::utils::BroadcastResult res =
        temper::utils::compute_broadcast({a_desc, b_desc, c_desc});

    const std::vector<uint64_t> out_shape = std::move(res.shape);
    const int64_t out_rank = static_cast<int64_t>(out_shape.size());

    const std::vector<uint64_t> x_bcast = std::move(res.strides[0]);
    const std::vector<uint64_t> loc_bcast = std::move(res.strides[1]);
    const std::vector<uint64_t> scale_bcast = std::move(res.strides[2]);

    Tensor<value_t> result(out_shape, x.get_memory_location());

    uint64_t* p_out_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));
    uint64_t* p_x_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));
    uint64_t* p_loc_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));
    uint64_t* p_scale_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));

    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    bool alloc_ok = (p_out_divs && p_x_bcast && p_loc_bcast &&
                     p_scale_bcast && p_error_flag);

    if (!alloc_ok)
    {
        sycl::free(p_out_divs, g_sycl_queue);
        sycl::free(p_x_bcast, g_sycl_queue);
        sycl::free(p_loc_bcast, g_sycl_queue);
        sycl::free(p_scale_bcast, g_sycl_queue);
        sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    const std::vector<uint64_t> out_divs = std::move(res.divisors);

    g_sycl_queue.memcpy(p_out_divs,
        out_divs.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();
    g_sycl_queue.memcpy(p_x_bcast,
        x_bcast.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();
    g_sycl_queue.memcpy(p_loc_bcast,
        loc_bcast.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();
    g_sycl_queue.memcpy(p_scale_bcast,
        scale_bcast.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();

    *p_error_flag = 0;

    const value_t* p_x = x.get_data();
    const value_t* p_loc = loc.get_data();
    const value_t* p_scale = scale.get_data();
    value_t* p_out = result.get_data();

    const double sqrt2 = std::sqrt(2.0);

    const uint64_t total_output_elems = result.get_num_elements();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);

            const uint64_t x_idx = temper::sycl_utils::idx_of(
                flat, p_out_divs, p_x_bcast, out_rank);
            const uint64_t loc_idx = temper::sycl_utils::idx_of(
                flat, p_out_divs, p_loc_bcast, out_rank);
            const uint64_t scale_idx = temper::sycl_utils::idx_of(
                flat, p_out_divs, p_scale_bcast, out_rank);

            double xp = static_cast<double>(p_x[x_idx]);
            double locp = static_cast<double>(p_loc[loc_idx]);
            double scalep = static_cast<double>(p_scale[scale_idx]);

            temper::sycl_utils::device_check_nan_and_set<double>
                (xp, p_error_flag);
            temper::sycl_utils::device_check_nan_and_set<double>
                (locp, p_error_flag);
            temper::sycl_utils::device_check_nan_and_set<double>
                (scalep, p_error_flag);

            if (scalep <= 0.0)
            {
                p_error_flag[0] = 4;
                p_out[flat] = std::numeric_limits<value_t>::quiet_NaN();
                return;
            }

            double z = (xp - locp) / (scalep * sqrt2);
            double erfv = sycl::erf(z);
            double outv = 0.5 * (1.0 + erfv);

            temper::sycl_utils::device_check_finite_and_set<double>
                (outv, p_error_flag);
            p_out[flat] = static_cast<value_t>(outv);
        });
    }).wait();

    int32_t err = *p_error_flag;

    sycl::free(p_out_divs, g_sycl_queue);
    sycl::free(p_x_bcast, g_sycl_queue);
    sycl::free(p_loc_bcast, g_sycl_queue);
    sycl::free(p_scale_bcast, g_sycl_queue);
    sycl::free(p_error_flag, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::invalid_argument(R"(norm::cdf:
                NaN detected in inputs.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(norm::cdf:
                non-finite result (overflow or Inf) produced.)");
        }
        if (err == 4)
        {
            throw std::invalid_argument(R"(norm::cdf:
                scale must be positive.)");
        }
        throw std::runtime_error(R"(norm::cdf:
            numeric error during cdf computation.)");
    }

    return result;
}
template Tensor<float> cdf<float>
(const Tensor<float>&, const Tensor<float>&, const Tensor<float>&);

template<typename value_t>
Tensor<value_t> ppf(const Tensor<value_t>& q,
    const Tensor<value_t>& loc,
    const Tensor<value_t>& scale)
{
    const std::vector<uint64_t> & q_shape = q.get_dimensions();
    const std::vector<uint64_t> & loc_shape = loc.get_dimensions();
    const std::vector<uint64_t> & scale_shape = scale.get_dimensions();

    if (q_shape.empty())
    {
        throw std::invalid_argument(R"(norm::ppf:
            q tensor has no elements.)");
    }
    if (loc_shape.empty())
    {
        throw std::invalid_argument(R"(norm::ppf:
            loc tensor has no elements.)");
    }
    if (scale_shape.empty())
    {
        throw std::invalid_argument(R"(norm::ppf:
            scale tensor has no elements.)");
    }

    temper::utils::TensorDesc a_desc{q_shape, q.get_strides()};
    temper::utils::TensorDesc b_desc{loc_shape, loc.get_strides()};
    temper::utils::TensorDesc c_desc{scale_shape, scale.get_strides()};

    temper::utils::BroadcastResult res =
        temper::utils::compute_broadcast({a_desc, b_desc, c_desc});

    const std::vector<uint64_t> out_shape = std::move(res.shape);
    const int64_t out_rank = static_cast<int64_t>(out_shape.size());

    /* per-operand broadcast-aware strides */
    const std::vector<uint64_t> q_bcast = std::move(res.strides[0]);
    const std::vector<uint64_t> loc_bcast = std::move(res.strides[1]);
    const std::vector<uint64_t> scale_bcast = std::move(res.strides[2]);

    Tensor<value_t> result(out_shape, q.get_memory_location());

    uint64_t* p_out_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));
    uint64_t* p_q_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));
    uint64_t* p_loc_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));
    uint64_t* p_scale_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));

    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    bool alloc_ok = (p_out_divs && p_q_bcast && p_loc_bcast &&
                     p_scale_bcast && p_error_flag);

    if (!alloc_ok)
    {
        sycl::free(p_out_divs, g_sycl_queue);
        sycl::free(p_q_bcast, g_sycl_queue);
        sycl::free(p_loc_bcast, g_sycl_queue);
        sycl::free(p_scale_bcast, g_sycl_queue);
        sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    const std::vector<uint64_t> out_divs = std::move(res.divisors);

    g_sycl_queue.memcpy(p_out_divs,
        out_divs.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();
    g_sycl_queue.memcpy(p_q_bcast,
        q_bcast.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();
    g_sycl_queue.memcpy(p_loc_bcast,
        loc_bcast.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();
    g_sycl_queue.memcpy(p_scale_bcast,
        scale_bcast.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();

    *p_error_flag = 0;

    const value_t* p_q = q.get_data();
    const value_t* p_loc = loc.get_data();
    const value_t* p_scale = scale.get_data();
    value_t* p_out = result.get_data();

    // Coefficients for Acklam's inverse normal approximation.
    const double a1 = -3.969683028665376e+01;
    const double a2 = 2.209460984245205e+02;
    const double a3 = -2.759285104469687e+02;
    const double a4 = 1.383577518672690e+02;
    const double a5 = -3.066479806614716e+01;
    const double a6 = 2.506628277459239e+00;

    const double b1 = -5.447609879822406e+01;
    const double b2 = 1.615858368580409e+02;
    const double b3 = -1.556989798598866e+02;
    const double b4 = 6.680131188771972e+01;
    const double b5 = -1.328068155288572e+01;

    const double c1 = -7.784894002430293e-03;
    const double c2 = -3.223964580411365e-01;
    const double c3 = -2.400758277161838e+00;
    const double c4 = -2.549732539343734e+00;
    const double c5 = 4.374664141464968e+00;
    const double c6 = 2.938163982698783e+00;

    const double d1 = 7.784695709041462e-03;
    const double d2 = 3.224671290700398e-01;
    const double d3 = 2.445134137142996e+00;
    const double d4 = 3.754408661907416e+00;

    const double plow = 0.02425;
    const double phigh = 1.0 - plow;

    const uint64_t total_output_elems = result.get_num_elements();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);

            const uint64_t q_idx = temper::sycl_utils::idx_of(
                flat, p_out_divs, p_q_bcast, out_rank);
            const uint64_t loc_idx = temper::sycl_utils::idx_of(
                flat, p_out_divs, p_loc_bcast, out_rank);
            const uint64_t scale_idx = temper::sycl_utils::idx_of(
                flat, p_out_divs, p_scale_bcast, out_rank);

            double qp = static_cast<double>(p_q[q_idx]);
            double locp = static_cast<double>(p_loc[loc_idx]);
            double scalep = static_cast<double>(p_scale[scale_idx]);

            temper::sycl_utils::device_check_nan_and_set<double>
                (qp, p_error_flag);
            temper::sycl_utils::device_check_nan_and_set<double>
                (locp, p_error_flag);
            temper::sycl_utils::device_check_nan_and_set<double>
                (scalep, p_error_flag);

            if (scalep <= 0.0)
            {
                p_error_flag[0] = 4;
                p_out[flat] = std::numeric_limits<value_t>::quiet_NaN();
                return;
            }

            if (!(qp >= 0.0 && qp <= 1.0))
            {
                p_error_flag[0] = 3;
                p_out[flat] = std::numeric_limits<value_t>::quiet_NaN();
                return;
            }

            double x;
            if (qp == 0.0)
            {
                x = -INFINITY;
            }
            else if (qp == 1.0)
            {
                x = INFINITY;
            }
            else if (qp < plow)
            {
                double ql = qp;
                double r = sycl::sqrt(-2.0 * sycl::log(ql));
                x = (((((c1 * r + c2) * r + c3) * r + c4) * r + c5) * r + c6) /
                    ((((d1 * r + d2) * r + d3) * r + d4) * r + 1.0);
                x = -x;
            }
            else if (qp > phigh)
            {
                double qh = 1.0 - qp;
                double r = sycl::sqrt(-2.0 * sycl::log(qh));
                x = (((((c1 * r + c2) * r + c3) * r + c4) * r + c5) * r + c6) /
                    ((((d1 * r + d2) * r + d3) * r + d4) * r + 1.0);
            }
            else
            {
                double q0 = qp - 0.5;
                double r = q0 * q0;
                double num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5)
                    * r + a6) * q0;
                double den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5)
                    * r + 1.0);
                x = num / den;
            }

            double outv = locp + scalep * x;
            temper::sycl_utils::device_check_finite_and_set<double>
                (outv, p_error_flag);
            p_out[flat] = static_cast<value_t>(outv);
        });
    }).wait();

    int32_t err = *p_error_flag;

    sycl::free(p_out_divs, g_sycl_queue);
    sycl::free(p_q_bcast, g_sycl_queue);
    sycl::free(p_loc_bcast, g_sycl_queue);
    sycl::free(p_scale_bcast, g_sycl_queue);
    sycl::free(p_error_flag, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::invalid_argument(R"(norm::ppf: NaN detected in inputs.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(norm::ppf:
                non-finite result (overflow or Inf) produced.)");
        }
        if (err == 3)
        {
            throw std::invalid_argument(R"(norm::ppf:
                q values must be in [0,1].)");
        }
        if (err == 4)
        {
            throw std::invalid_argument(R"(norm::ppf: scale must be positive.)");
        }
        throw std::runtime_error(R"(norm::ppf:
            numeric error during ppf computation.)");
    }

    return result;
}
template Tensor<float> ppf<float>
(const Tensor<float>&, const Tensor<float>&, const Tensor<float>&);

template<typename value_t>
Tensor<value_t> isf(const Tensor<value_t>& q,
    const Tensor<value_t>& loc,
    const Tensor<value_t>& scale)
{
    const std::vector<uint64_t> & q_shape = q.get_dimensions();
    const std::vector<uint64_t> & loc_shape = loc.get_dimensions();
    const std::vector<uint64_t> & scale_shape = scale.get_dimensions();

    if (q_shape.empty())
    {
        throw std::invalid_argument(R"(norm::isf:
            q tensor has no elements.)");
    }
    if (loc_shape.empty())
    {
        throw std::invalid_argument(R"(norm::isf:
            loc tensor has no elements.)");
    }
    if (scale_shape.empty())
    {
        throw std::invalid_argument(R"(norm::isf:
            scale tensor has no elements.)");
    }

    temper::utils::TensorDesc a_desc{q_shape, q.get_strides()};
    temper::utils::TensorDesc b_desc{loc_shape, loc.get_strides()};
    temper::utils::TensorDesc c_desc{scale_shape, scale.get_strides()};

    temper::utils::BroadcastResult res =
        temper::utils::compute_broadcast({a_desc, b_desc, c_desc});

    const std::vector<uint64_t> out_shape = std::move(res.shape);
    const int64_t out_rank = static_cast<int64_t>(out_shape.size());

    const std::vector<uint64_t> q_bcast = std::move(res.strides[0]);
    const std::vector<uint64_t> loc_bcast = std::move(res.strides[1]);
    const std::vector<uint64_t> scale_bcast = std::move(res.strides[2]);

    Tensor<value_t> result(out_shape, q.get_memory_location());

    uint64_t* p_out_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));
    uint64_t* p_q_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));
    uint64_t* p_loc_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));
    uint64_t* p_scale_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));

    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    bool alloc_ok = (p_out_divs && p_q_bcast && p_loc_bcast &&
                     p_scale_bcast && p_error_flag);

    if (!alloc_ok)
    {
        sycl::free(p_out_divs, g_sycl_queue);
        sycl::free(p_q_bcast, g_sycl_queue);
        sycl::free(p_loc_bcast, g_sycl_queue);
        sycl::free(p_scale_bcast, g_sycl_queue);
        sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    const std::vector<uint64_t> out_divs = std::move(res.divisors);

    g_sycl_queue.memcpy(p_out_divs,
        out_divs.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();
    g_sycl_queue.memcpy(p_q_bcast,
        q_bcast.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();
    g_sycl_queue.memcpy(p_loc_bcast,
        loc_bcast.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();
    g_sycl_queue.memcpy(p_scale_bcast,
        scale_bcast.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();

    *p_error_flag = 0;

    const value_t* p_q = q.get_data();
    const value_t* p_loc = loc.get_data();
    const value_t* p_scale = scale.get_data();
    value_t* p_out = result.get_data();

    // Coefficients for Acklam's inverse normal approximation (same as ppf)
    const double a1 = -3.969683028665376e+01;
    const double a2 = 2.209460984245205e+02;
    const double a3 = -2.759285104469687e+02;
    const double a4 = 1.383577518672690e+02;
    const double a5 = -3.066479806614716e+01;
    const double a6 = 2.506628277459239e+00;

    const double b1 = -5.447609879822406e+01;
    const double b2 = 1.615858368580409e+02;
    const double b3 = -1.556989798598866e+02;
    const double b4 = 6.680131188771972e+01;
    const double b5 = -1.328068155288572e+01;

    const double c1 = -7.784894002430293e-03;
    const double c2 = -3.223964580411365e-01;
    const double c3 = -2.400758277161838e+00;
    const double c4 = -2.549732539343734e+00;
    const double c5 = 4.374664141464968e+00;
    const double c6 = 2.938163982698783e+00;

    const double d1 = 7.784695709041462e-03;
    const double d2 = 3.224671290700398e-01;
    const double d3 = 2.445134137142996e+00;
    const double d4 = 3.754408661907416e+00;

    const double plow = 0.02425;
    const double phigh = 1.0 - plow;

    const uint64_t total_output_elems = result.get_num_elements();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);

            const uint64_t q_idx = temper::sycl_utils::idx_of(
                flat, p_out_divs, p_q_bcast, out_rank);
            const uint64_t loc_idx = temper::sycl_utils::idx_of(
                flat, p_out_divs, p_loc_bcast, out_rank);
            const uint64_t scale_idx = temper::sycl_utils::idx_of(
                flat, p_out_divs, p_scale_bcast, out_rank);

            double qp = static_cast<double>(p_q[q_idx]);
            double locp = static_cast<double>(p_loc[loc_idx]);
            double scalep = static_cast<double>(p_scale[scale_idx]);

            temper::sycl_utils::device_check_nan_and_set<double>
                (qp, p_error_flag);
            temper::sycl_utils::device_check_nan_and_set<double>
                (locp, p_error_flag);
            temper::sycl_utils::device_check_nan_and_set<double>
                (scalep, p_error_flag);

            if (scalep <= 0.0)
            {
                p_error_flag[0] = 4;
                p_out[flat] = std::numeric_limits<value_t>::quiet_NaN();
                return;
            }

            if (!(qp >= 0.0 && qp <= 1.0))
            {
                p_error_flag[0] = 3;
                p_out[flat] = std::numeric_limits<value_t>::quiet_NaN();
                return;
            }

            // complement inside kernel: isf(q) == ppf(1 - q)
            double qcomp = 1.0 - qp;

            double x;
            if (qcomp == 0.0)
            {
                x = -INFINITY;
            }
            else if (qcomp == 1.0)
            {
                x = INFINITY;
            }
            else if (qcomp < plow)
            {
                double r = sycl::sqrt(-2.0 * sycl::log(qcomp));
                x = (((((c1 * r + c2) * r + c3) * r + c4) * r + c5) * r + c6) /
                    ((((d1 * r + d2) * r + d3) * r + d4) * r + 1.0);
                x = -x;
            }
            else if (qcomp > phigh)
            {
                double qh = 1.0 - qcomp;
                double r = sycl::sqrt(-2.0 * sycl::log(qh));
                x = (((((c1 * r + c2) * r + c3) * r + c4) * r + c5) * r + c6) /
                    ((((d1 * r + d2) * r + d3) * r + d4) * r + 1.0);
            }
            else
            {
                double q0 = qcomp - 0.5;
                double r = q0 * q0;
                double num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5)
                    * r + a6) * q0;
                double den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5)
                    * r + 1.0);
                x = num / den;
            }

            double outv = locp + scalep * x;
            temper::sycl_utils::device_check_finite_and_set<double>
                (outv, p_error_flag);
            p_out[flat] = static_cast<value_t>(outv);
        });
    }).wait();

    int32_t err = *p_error_flag;

    sycl::free(p_out_divs, g_sycl_queue);
    sycl::free(p_q_bcast, g_sycl_queue);
    sycl::free(p_loc_bcast, g_sycl_queue);
    sycl::free(p_scale_bcast, g_sycl_queue);
    sycl::free(p_error_flag, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::invalid_argument(R"(norm::isf: NaN detected in inputs.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(norm::isf:
                non-finite result (overflow or Inf) produced.)");
        }
        if (err == 3)
        {
            throw std::invalid_argument(R"(norm::isf:
                q values must be in [0,1].)");
        }
        if (err == 4)
        {
            throw std::invalid_argument(R"(norm::isf: scale must be positive.)");
        }
        throw std::runtime_error(R"(norm::isf:
            numeric error during isf computation.)");
    }

    return result;
}
template Tensor<float> isf<float>
(const Tensor<float>&, const Tensor<float>&, const Tensor<float>&);


template<typename value_t>
Tensor<value_t> rvs(const Tensor<value_t>& loc,
    const Tensor<value_t>& scale,
    const std::vector<uint64_t>& out_shape,
    MemoryLocation res_loc,
    uint64_t seed)
{
    const std::vector<uint64_t> & loc_shape = loc.get_dimensions();
    const std::vector<uint64_t> & scale_shape = scale.get_dimensions();

    if (loc_shape.empty())
    {
        throw std::invalid_argument(R"(norm::rvs:
            loc tensor has no elements.)");
    }
    if (scale_shape.empty())
    {
        throw std::invalid_argument(R"(norm::rvs:
            scale tensor has no elements.)");
    }

    Tensor<value_t> q(out_shape, res_loc);
    const uint64_t total_output_elems = q.get_num_elements();

    const int64_t out_rank = static_cast<int64_t>(out_shape.size());
    std::vector<uint64_t> out_divs = temper::utils::compute_divisors(out_shape);
    uint64_t* p_out_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * out_rank, g_sycl_queue));

    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    bool alloc_ok = (p_out_divs && p_error_flag);
    if (!alloc_ok)
    {
        sycl::free(p_out_divs, g_sycl_queue);
        sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    g_sycl_queue.memcpy(p_out_divs,
        out_divs.data(), sizeof(uint64_t) * out_rank).wait();
    *p_error_flag = 0;

    value_t* p_q = q.get_data();

    if (seed == 0ULL)
    {
        std::random_device rd;
        seed = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
        if (seed == 0ULL) seed = 0x9e3779b97f4a7c15ULL;
    }

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);

            uint64_t s = seed ^ (flat + 0x9e3779b97f4a7c15ULL);
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            uint64_t rnd = s * 2685821657736338717ULL;

            double u = static_cast<double>(rnd) / 18446744073709551616.0;

            if (u < 1e-16) u = 1e-16;
            if (u > 1.0 - 1e-16) u = 1.0 - 1e-16;

            if (!(u >= 0.0 && u < 1.0))
            {
                p_error_flag[0] = 1;
                p_q[flat] = std::numeric_limits<value_t>::quiet_NaN();
                return;
            }

            p_q[flat] = static_cast<value_t>(u);
        });
    }).wait();

    int32_t err = *p_error_flag;
    sycl::free(p_out_divs, g_sycl_queue);
    sycl::free(p_error_flag, g_sycl_queue);

    if (err != 0)
    {
        throw std::runtime_error(R"(norm::rvs:
            numeric error during uniform generation.)");
    }

    Tensor<value_t> result = ppf(q, loc, scale);

    return result;
}
template Tensor<float> rvs<float>(const Tensor<float>&, const Tensor<float>&,
    const std::vector<uint64_t>&, MemoryLocation, uint64_t);

template<typename value_t>
Tensor<value_t> mean(const Tensor<value_t>& loc,
    const Tensor<value_t>& scale)
{
    (void) scale;
    return loc;
}
template Tensor<float> mean<float>
(const Tensor<float>&, const Tensor<float>&);

template<typename value_t>
Tensor<value_t> var(const Tensor<value_t>& loc,
    const Tensor<value_t>& scale)
{
    (void) loc;
    return math::pow(scale, Tensor<value_t>(2));
}
template Tensor<float> var<float>
(const Tensor<float>&, const Tensor<float>&);

template<typename value_t>
Tensor<value_t> stddev(const Tensor<value_t>& loc,
    const Tensor<value_t>& scale)
{
    (void) loc;
    return scale;
}
template Tensor<float> stddev<float>
(const Tensor<float>&, const Tensor<float>&);

} // namespace norm

namespace chisquare
{

template<typename value_t>
Tensor<value_t> pdf(const Tensor<value_t>& x,
    const Tensor<value_t>& k)
{
    const std::vector<uint64_t> & x_shape = x.get_dimensions();
    const std::vector<uint64_t> & k_shape = k.get_dimensions();

    if (x_shape.empty())
    {
        throw std::invalid_argument(R"(chisquare::pdf:
            x tensor has no elements.)");
    }
    if (k_shape.empty())
    {
        throw std::invalid_argument(R"(chisquare::pdf:
            k tensor has no elements.)");
    }

    temper::utils::TensorDesc x_desc{x_shape, x.get_strides()};
    temper::utils::TensorDesc k_desc{k_shape, k.get_strides()};

    temper::utils::BroadcastResult res =
        temper::utils::compute_broadcast({x_desc, k_desc});

    const std::vector<uint64_t> out_shape = std::move(res.shape);
    const int64_t out_rank = static_cast<int64_t>(out_shape.size());

    const std::vector<uint64_t> x_bcast = std::move(res.strides[0]);
    const std::vector<uint64_t> k_bcast = std::move(res.strides[1]);

    Tensor<value_t> result(out_shape, x.get_memory_location());

    uint64_t* p_out_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));
    uint64_t* p_x_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));
    uint64_t* p_k_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));

    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    bool alloc_ok = (p_out_divs && p_x_bcast && p_k_bcast && p_error_flag);

    if (!alloc_ok)
    {
        sycl::free(p_out_divs, g_sycl_queue);
        sycl::free(p_x_bcast, g_sycl_queue);
        sycl::free(p_k_bcast, g_sycl_queue);
        sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    const std::vector<uint64_t> out_divs = std::move(res.divisors);

    g_sycl_queue.memcpy(p_out_divs,
        out_divs.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();
    g_sycl_queue.memcpy(p_x_bcast,
        x_bcast.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();
    g_sycl_queue.memcpy(p_k_bcast,
        k_bcast.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();

    *p_error_flag = 0;

    const value_t* p_x = x.get_data();
    const value_t* p_k = k.get_data();
    value_t* p_out = result.get_data();

    const uint64_t total_output_elems = result.get_num_elements();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);

            const uint64_t x_idx = temper::sycl_utils::idx_of(
                flat, p_out_divs, p_x_bcast, out_rank);
            const uint64_t k_idx = temper::sycl_utils::idx_of(
                flat, p_out_divs, p_k_bcast, out_rank);

            double xp = static_cast<double>(p_x[x_idx]);
            double kp = static_cast<double>(p_k[k_idx]);

            temper::sycl_utils::device_check_nan_and_set<double>
                (xp, p_error_flag);
            temper::sycl_utils::device_check_nan_and_set<double>
                (kp, p_error_flag);

            if (kp <= 0.0)
            {
                p_error_flag[0] = 3;
                p_out[flat] = std::numeric_limits<value_t>::quiet_NaN();
                return;
            }

            if (xp < 0.0)
            {
                p_error_flag[0] = 4;
                p_out[flat] = std::numeric_limits<value_t>::quiet_NaN();
                return;
            }

            double denom = sycl::pow(2.0, kp * 0.5) * sycl::tgamma(kp * 0.5);
            double outv = (1.0 / denom) * sycl::pow(xp, (kp * 0.5 - 1.0)) *
                sycl::exp(-xp * 0.5); 

            temper::sycl_utils::device_check_finite_and_set<double>
                (outv, p_error_flag);
            p_out[flat] = static_cast<value_t>(outv);
        });
    }).wait();

    int32_t err = *p_error_flag;

    sycl::free(p_out_divs, g_sycl_queue);
    sycl::free(p_x_bcast, g_sycl_queue);
    sycl::free(p_k_bcast, g_sycl_queue);
    sycl::free(p_error_flag, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::invalid_argument(R"(chisquare::pdf:
                NaN detected in inputs.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(chisquare::pdf:
                non-finite result (overflow or Inf) produced.)");
        }
        if (err == 3)
        {
            throw std::invalid_argument(R"(chisquare::pdf:
                k must be positive.)");
        }
        if (err == 4)
        {
            throw std::invalid_argument(R"(chisquare::pdf:
                x must be positive.)");
        }
        throw std::runtime_error(R"(chisquare::pdf:
            numeric error during pdf computation.)");
    }

    return result;
}
template Tensor<float> pdf<float>
(const Tensor<float>&, const Tensor<float>&);

template<typename value_t>
Tensor<value_t> cdf(const Tensor<value_t>& x,
    const Tensor<value_t>& k)
{
    const std::vector<uint64_t> & x_shape = x.get_dimensions();
    const std::vector<uint64_t> & k_shape = k.get_dimensions();

    if (x_shape.empty())
    {
        throw std::invalid_argument(R"(chisquare::cdf:
            x tensor has no elements.)");
    }
    if (k_shape.empty())
    {
        throw std::invalid_argument(R"(chisquare::cdf:
            k tensor has no elements.)");
    }

    temper::utils::TensorDesc x_desc{x_shape, x.get_strides()};
    temper::utils::TensorDesc k_desc{k_shape, k.get_strides()};

    temper::utils::BroadcastResult res =
        temper::utils::compute_broadcast({x_desc, k_desc});

    const std::vector<uint64_t> out_shape = std::move(res.shape);
    const int64_t out_rank = static_cast<int64_t>(out_shape.size());

    const std::vector<uint64_t> x_bcast = std::move(res.strides[0]);
    const std::vector<uint64_t> k_bcast = std::move(res.strides[1]);

    Tensor<value_t> result(out_shape, x.get_memory_location());

    uint64_t* p_out_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));
    uint64_t* p_x_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));
    uint64_t* p_k_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));

    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    bool alloc_ok = (p_out_divs && p_x_bcast && p_k_bcast && p_error_flag);

    if (!alloc_ok)
    {
        sycl::free(p_out_divs, g_sycl_queue);
        sycl::free(p_x_bcast, g_sycl_queue);
        sycl::free(p_k_bcast, g_sycl_queue);
        sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    const std::vector<uint64_t> out_divs = std::move(res.divisors);

    g_sycl_queue.memcpy(p_out_divs,
        out_divs.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();
    g_sycl_queue.memcpy(p_x_bcast,
        x_bcast.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();
    g_sycl_queue.memcpy(p_k_bcast,
        k_bcast.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();

    *p_error_flag = 0;

    const value_t* p_x = x.get_data();
    const value_t* p_k = k.get_data();
    value_t* p_out = result.get_data();

    const uint64_t total_output_elems = result.get_num_elements();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);

            const uint64_t x_idx = temper::sycl_utils::idx_of(
                flat, p_out_divs, p_x_bcast, out_rank);
            const uint64_t k_idx = temper::sycl_utils::idx_of(
                flat, p_out_divs, p_k_bcast, out_rank);

            double xp = static_cast<double>(p_x[x_idx]);
            double kp = static_cast<double>(p_k[k_idx]);

            temper::sycl_utils::device_check_nan_and_set<double>
                (xp, p_error_flag);
            temper::sycl_utils::device_check_nan_and_set<double>
                (kp, p_error_flag);

            if (kp <= 0.0)
            {
                p_error_flag[0] = 3;
                p_out[flat] = std::numeric_limits<value_t>::quiet_NaN();
                return;
            }

            if (xp < 0.0)
            {
                p_error_flag[0] = 4;
                p_out[flat] = std::numeric_limits<value_t>::quiet_NaN();
                return;
            }

            double outv = sycl_utils::regularized_gamma(kp / 2, xp / 2);

            temper::sycl_utils::device_check_finite_and_set<double>
                (outv, p_error_flag);
            p_out[flat] = static_cast<value_t>(outv);
        });
    }).wait();

    int32_t err = *p_error_flag;

    sycl::free(p_out_divs, g_sycl_queue);
    sycl::free(p_x_bcast, g_sycl_queue);
    sycl::free(p_k_bcast, g_sycl_queue);
    sycl::free(p_error_flag, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::invalid_argument(R"(chisquare::cdf:
                NaN detected in inputs.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(chisquare::cdf:
                non-finite result (overflow or Inf) produced.)");
        }
        if (err == 3)
        {
            throw std::invalid_argument(R"(chisquare::cdf:
                k must be positive.)");
        }
        if (err == 4)
        {
            throw std::invalid_argument(R"(chisquare::cdf:
                x must be positive.)");
        }
        throw std::runtime_error(R"(chisquare::cdf:
            numeric error during pdf computation.)");
    }

    return result;
}
template Tensor<float> cdf<float>
(const Tensor<float>&, const Tensor<float>&);

template<typename value_t>
Tensor<value_t> ppf(const Tensor<value_t>& q,
    const Tensor<value_t>& k)
{
    const std::vector<uint64_t> & q_shape = q.get_dimensions();
    const std::vector<uint64_t> & k_shape = k.get_dimensions();

    if (q_shape.empty())
    {
        throw std::invalid_argument(R"(chisquare::ppf:
            q tensor has no elements.)");
    }
    if (k_shape.empty())
    {
        throw std::invalid_argument(R"(chisquare::ppf:
            k tensor has no elements.)");
    }

    temper::utils::TensorDesc q_desc{q_shape, q.get_strides()};
    temper::utils::TensorDesc k_desc{k_shape, k.get_strides()};

    temper::utils::BroadcastResult res =
        temper::utils::compute_broadcast({q_desc, k_desc});

    const std::vector<uint64_t> out_shape = std::move(res.shape);
    const int64_t out_rank = static_cast<int64_t>(out_shape.size());

    const std::vector<uint64_t> q_bcast = std::move(res.strides[0]);
    const std::vector<uint64_t> k_bcast = std::move(res.strides[1]);

    Tensor<value_t> result(out_shape, q.get_memory_location());

    uint64_t* p_out_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));
    uint64_t* p_q_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));
    uint64_t* p_k_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * static_cast<size_t>(out_rank),
                            g_sycl_queue));

    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    bool alloc_ok = (p_out_divs && p_q_bcast && p_k_bcast && p_error_flag);

    if (!alloc_ok)
    {
        sycl::free(p_out_divs, g_sycl_queue);
        sycl::free(p_q_bcast, g_sycl_queue);
        sycl::free(p_k_bcast, g_sycl_queue);
        sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    const std::vector<uint64_t> out_divs = std::move(res.divisors);

    g_sycl_queue.memcpy(p_out_divs,
        out_divs.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();
    g_sycl_queue.memcpy(p_q_bcast,
        q_bcast.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();
    g_sycl_queue.memcpy(p_k_bcast,
        k_bcast.data(), sizeof(uint64_t) * static_cast<size_t>(out_rank))
        .wait();

    *p_error_flag = 0;

    const value_t* p_q = q.get_data();
    const value_t* p_k = k.get_data();
    value_t* p_out = result.get_data();

    const uint64_t total_output_elems = result.get_num_elements();

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);

            const uint64_t q_idx = temper::sycl_utils::idx_of(
                flat, p_out_divs, p_q_bcast, out_rank);
            const uint64_t k_idx = temper::sycl_utils::idx_of(
                flat, p_out_divs, p_k_bcast, out_rank);

            double qp = static_cast<double>(p_q[q_idx]);
            double kp = static_cast<double>(p_k[k_idx]);

            temper::sycl_utils::device_check_nan_and_set<double>
                (qp, p_error_flag);
            temper::sycl_utils::device_check_nan_and_set<double>
                (kp, p_error_flag);

            if (kp <= 0.0)
            {
                p_error_flag[0] = 3;
                p_out[flat] = std::numeric_limits<value_t>::quiet_NaN();
                return;
            }

            if (qp < 0.0 || qp > 1.0)
            {
                p_error_flag[0] = 4;
                p_out[flat] = std::numeric_limits<value_t>::quiet_NaN();
                return;
            }

            double outv = 2.0 * sycl_utils::inverse_regularized_gamma
                (kp / 2.0, qp);

            temper::sycl_utils::device_check_finite_and_set<double>
                (outv, p_error_flag);
            p_out[flat] = static_cast<value_t>(outv);
        });
    }).wait();

    int32_t err = *p_error_flag;

    sycl::free(p_out_divs, g_sycl_queue);
    sycl::free(p_q_bcast, g_sycl_queue);
    sycl::free(p_k_bcast, g_sycl_queue);
    sycl::free(p_error_flag, g_sycl_queue);

    if (err != 0)
    {
        if (err == 1)
        {
            throw std::invalid_argument(R"(chisquare::ppf:
                NaN detected in inputs.)");
        }
        if (err == 2)
        {
            throw std::runtime_error(R"(chisquare::ppf:
                non-finite result (overflow or Inf) produced.)");
        }
        if (err == 3)
        {
            throw std::invalid_argument(R"(chisquare::ppf:
                k must be positive.)");
        }
        if (err == 4)
        {
            throw std::invalid_argument(R"(chisquare::ppf:
                q values must be in [0,1].)");
        }
        throw std::runtime_error(R"(chisquare::ppf:
            numeric error during pdf computation.)");
    }

    return result;
}
template Tensor<float> ppf<float>
(const Tensor<float>&, const Tensor<float>&);

template<typename value_t>
Tensor<value_t> isf(const Tensor<value_t>& q,
    const Tensor<value_t>& k)
{
    const std::vector<uint64_t> & q_shape = q.get_dimensions();
    const std::vector<uint64_t> & k_shape = k.get_dimensions();

    if (q_shape.empty())
    {
        throw std::invalid_argument(R"(chisquare::isf:
            q tensor has no elements.)");
    }
    if (k_shape.empty())
    {
        throw std::invalid_argument(R"(chisquare::isf:
            k tensor has no elements.)");
    }
    return ppf((Tensor<value_t>(static_cast<value_t>(1)) - q), k);
}
template Tensor<float> isf<float>
(const Tensor<float>&, const Tensor<float>&);

template<typename value_t>
Tensor<value_t> rvs(const Tensor<value_t>& k,
    const std::vector<uint64_t>& out_shape,
    MemoryLocation res_loc,
    uint64_t seed)
{
    const std::vector<uint64_t> & k_shape = k.get_dimensions();

    if (k_shape.empty())
    {
        throw std::invalid_argument(R"(chisquare::rvs:
            k tensor has no elements.)");
    }

    Tensor<value_t> q(out_shape, res_loc);
    const uint64_t total_output_elems = q.get_num_elements();

    const int64_t out_rank = static_cast<int64_t>(out_shape.size());
    std::vector<uint64_t> out_divs = temper::utils::compute_divisors(out_shape);
    uint64_t* p_out_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * out_rank, g_sycl_queue));

    int32_t* p_error_flag = static_cast<int32_t*>(
        sycl::malloc_shared(sizeof(int32_t), g_sycl_queue));

    bool alloc_ok = (p_out_divs && p_error_flag);
    if (!alloc_ok)
    {
        sycl::free(p_out_divs, g_sycl_queue);
        sycl::free(p_error_flag, g_sycl_queue);
        throw std::bad_alloc();
    }

    g_sycl_queue.memcpy(p_out_divs,
        out_divs.data(), sizeof(uint64_t) * out_rank).wait();
    *p_error_flag = 0;

    value_t* p_q = q.get_data();

    if (seed == 0ULL)
    {
        std::random_device rd;
        seed = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
        if (seed == 0ULL) seed = 0x9e3779b97f4a7c15ULL;
    }

    g_sycl_queue.submit([&](sycl::handler& cgh)
    {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_output_elems)),
            [=](sycl::id<1> id)
        {
            const uint64_t flat = static_cast<uint64_t>(id[0]);

            uint64_t s = seed ^ (flat + 0x9e3779b97f4a7c15ULL);
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            uint64_t rnd = s * 2685821657736338717ULL;

            double u = static_cast<double>(rnd) / 18446744073709551616.0;

            if (u < 1e-16) u = 1e-16;
            if (u > 1.0 - 1e-16) u = 1.0 - 1e-16;

            if (!(u >= 0.0 && u < 1.0))
            {
                p_error_flag[0] = 1;
                p_q[flat] = std::numeric_limits<value_t>::quiet_NaN();
                return;
            }

            p_q[flat] = static_cast<value_t>(u);
        });
    }).wait();

    int32_t err = *p_error_flag;
    sycl::free(p_out_divs, g_sycl_queue);
    sycl::free(p_error_flag, g_sycl_queue);

    if (err != 0)
    {
        throw std::runtime_error(R"(chisquare::rvs:
            numeric error during uniform generation.)");
    }

    return ppf(q, k);
}
template Tensor<float> rvs<float>(const Tensor<float>&,
const std::vector<uint64_t>&, MemoryLocation, uint64_t);

} // namespace chisquare

} // namespace temper::stats