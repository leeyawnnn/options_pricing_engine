#pragma once

// Internal (non-installed) helper shared by black_scholes.cpp and greeks.cpp.
// It lives under src/ rather than include/ because it is an implementation
// detail of the library, not part of its public API.

#include <cmath>

#include "opt/market_data.hpp"

namespace opt::detail {

/// Intermediate Black-Scholes-Merton quantities, computed once and reused by
/// both the pricer and every Greek.
struct BsTerms {
    double disc_r;     ///< Risk-free discount factor, \f$e^{-rT}\f$.
    double disc_q;     ///< Dividend discount factor, \f$e^{-qT}\f$.
    double forward;    ///< Dividend-adjusted spot, \f$S e^{-qT}\f$.
    double pv_strike;  ///< Discounted strike, \f$K e^{-rT}\f$.
    double std_dev;    ///< Total volatility over the life, \f$\sigma\sqrt{T}\f$.
    double d1;         ///< First Black-Scholes argument.
    double d2;         ///< Second Black-Scholes argument.
    bool degenerate;   ///< True when \f$\sigma\sqrt{T} = 0\f$ or spot \f$\le 0\f$.
};

[[nodiscard]] inline BsTerms compute_terms(const MarketData& market, double strike,
                                           double expiry) noexcept {
    const double disc_r = std::exp(-market.rate * expiry);
    const double disc_q = std::exp(-market.dividend * expiry);
    const double forward = market.spot * disc_q;
    const double pv_strike = strike * disc_r;
    const double std_dev = market.volatility * std::sqrt(expiry);

    // No diffusion (zero vol or zero time) or a worthless underlying: d1/d2 are
    // undefined, so flag it and let each caller fall back to a finite limit.
    if (std_dev <= 0.0 || market.spot <= 0.0) {
        return {disc_r, disc_q, forward, pv_strike, std_dev, 0.0, 0.0, /*degenerate=*/true};
    }

    const double d1 = std::log(market.spot / strike) / std_dev +
                      (market.rate - market.dividend) * expiry / std_dev + 0.5 * std_dev;
    const double d2 = d1 - std_dev;
    return {disc_r, disc_q, forward, pv_strike, std_dev, d1, d2, /*degenerate=*/false};
}

}  // namespace opt::detail
