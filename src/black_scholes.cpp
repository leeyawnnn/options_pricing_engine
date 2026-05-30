#include "opt/black_scholes.hpp"

#include <algorithm>

#include "bs_internal.hpp"
#include "opt/normal_distribution.hpp"

namespace opt {

double black_scholes_call(const MarketData& market, double strike, double expiry) noexcept {
    const detail::BsTerms t = detail::compute_terms(market, strike, expiry);
    if (t.degenerate) {
        return std::max(t.forward - t.pv_strike, 0.0);
    }
    return t.forward * cumulative_normal(t.d1) - t.pv_strike * cumulative_normal(t.d2);
}

double black_scholes_put(const MarketData& market, double strike, double expiry) noexcept {
    const detail::BsTerms t = detail::compute_terms(market, strike, expiry);
    if (t.degenerate) {
        return std::max(t.pv_strike - t.forward, 0.0);
    }
    return t.pv_strike * cumulative_normal(-t.d2) - t.forward * cumulative_normal(-t.d1);
}

double black_scholes_price(const MarketData& market, double strike, double expiry,
                           OptionType type) noexcept {
    return type == OptionType::Call ? black_scholes_call(market, strike, expiry)
                                    : black_scholes_put(market, strike, expiry);
}

double put_call_parity_residual(const MarketData& market, double strike, double expiry) noexcept {
    const detail::BsTerms t = detail::compute_terms(market, strike, expiry);
    const double call = black_scholes_call(market, strike, expiry);
    const double put = black_scholes_put(market, strike, expiry);
    return (call - put) - (t.forward - t.pv_strike);
}

}  // namespace opt
