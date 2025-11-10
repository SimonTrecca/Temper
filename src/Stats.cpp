/**
 * @file Stats.cpp
 * @brief Statistical distributions utilities definitions.
 */

#include "temper/Stats.hpp"
#include "temper/Utils.hpp"
#include "temper/SYCLUtils.hpp"
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
Tensor<value_t> ppf(const Tensor<value_t>& q,
	const Tensor<value_t>& loc,
	const Tensor<value_t>& scale)
{
    const std::vector<uint64_t> q_shape = q.get_dimensions();
    const std::vector<uint64_t> loc_shape = loc.get_dimensions();
    const std::vector<uint64_t> scale_shape = scale.get_dimensions();

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

    temper::utils::TensorDesc a_desc{q_shape, q.get_strides(), {}};
    temper::utils::TensorDesc b_desc{loc_shape, loc.get_strides(), {}};
    temper::utils::TensorDesc c_desc{scale_shape, scale.get_strides(), {}};

    const int64_t full_rank = static_cast<int64_t>(std::max({a_desc.shape.size(),
    	b_desc.shape.size(), c_desc.shape.size()}));
    temper::utils::TensorDesc a_al =
    	temper::utils::align_tensor(a_desc, full_rank);
    temper::utils::TensorDesc b_al =
    	temper::utils::align_tensor(b_desc, full_rank);
    temper::utils::TensorDesc c_al =
    	temper::utils::align_tensor(c_desc, full_rank);

    temper::utils::BroadcastResult ab =
    	temper::utils::compute_broadcast(a_al, b_al);

    temper::utils::TensorDesc ab_owner;
    ab_owner.shape = ab.out.shape;
    ab_owner.strides = temper::utils::compute_divisors(ab.out.shape);

    temper::utils::BroadcastResult abc =
    	temper::utils::compute_broadcast(ab_owner, c_al);

    const std::vector<uint64_t> out_shape = abc.out.shape;
    const int64_t out_rank = static_cast<int64_t>(out_shape.size());

    std::vector<uint64_t> q_bcast(out_rank, 0);
    std::vector<uint64_t> loc_bcast(out_rank, 0);
    std::vector<uint64_t> scale_bcast = abc.b_strides;

    for (int64_t d = 0; d < out_rank; ++d)
	{
	    if (abc.a_strides[d] == 0)
	    {
	        q_bcast[d] = 0;
	        loc_bcast[d] = 0;
	    }
	    else
	    {
	        if (d < static_cast<int64_t>(ab.a_strides.size()))
	        {
	            q_bcast[d] = ab.a_strides[d];
	        }
	        else
	        {
	            q_bcast[d] = 0;
	        }

	        if (d < static_cast<int64_t>(ab.a_strides.size()))
	        {
	            loc_bcast[d] = ab.b_strides[d];
	        }
	        else
	        {
	            loc_bcast[d] = 0;
	        }
	    }
	}

    Tensor<value_t> result(out_shape, q.get_memory_location());

    uint64_t* p_out_divs = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * out_rank, g_sycl_queue));
    uint64_t* p_q_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * out_rank, g_sycl_queue));
    uint64_t* p_loc_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * out_rank, g_sycl_queue));
    uint64_t* p_scale_bcast = static_cast<uint64_t*>(
        sycl::malloc_device(sizeof(uint64_t) * out_rank, g_sycl_queue));

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

    std::vector<uint64_t> out_divs = abc.out.divisors;

    g_sycl_queue.memcpy(p_out_divs,
    	out_divs.data(), sizeof(uint64_t) * out_rank).wait();
    g_sycl_queue.memcpy(p_q_bcast,
    	q_bcast.data(), sizeof(uint64_t) * out_rank).wait();
    g_sycl_queue.memcpy(p_loc_bcast,
    	loc_bcast.data(), sizeof(uint64_t) * out_rank).wait();
    g_sycl_queue.memcpy(p_scale_bcast,
    	scale_bcast.data(), sizeof(uint64_t) * out_rank).wait();

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

            const uint64_t q_idx = temper::sycl_utils::idx_of
            	(flat, p_out_divs, p_q_bcast, out_rank);
            const uint64_t loc_idx = temper::sycl_utils::idx_of
            	(flat, p_out_divs, p_loc_bcast, out_rank);
            const uint64_t scale_idx = temper::sycl_utils::idx_of
            	(flat, p_out_divs, p_scale_bcast, out_rank);

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
            throw std::runtime_error(R"(norm::ppf: NaN detected in inputs.)");
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
Tensor<value_t> rvs(const Tensor<value_t>& loc,
    const Tensor<value_t>& scale,
    const std::vector<uint64_t>& out_shape,
    MemoryLocation res_loc,
    uint64_t seed)
{
    const std::vector<uint64_t> loc_shape = loc.get_dimensions();
    const std::vector<uint64_t> scale_shape = scale.get_dimensions();

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
                p_error_flag[0] = 2;
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
        if (err == 1)
        {
            throw std::runtime_error(R"(norm::rvs:
                NaN detected while generating uniforms.)");
        }
        throw std::runtime_error(R"(norm::rvs:
            numeric error during uniform generation.)");
    }

    Tensor<value_t> result = ppf(q, loc, scale);

    return result;
}
template Tensor<float> rvs<float>(const Tensor<float>&, const Tensor<float>&,
    const std::vector<uint64_t>&, MemoryLocation, uint64_t);

} // namespace norm

} // namespace temper::stats