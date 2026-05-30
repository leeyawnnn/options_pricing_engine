#include "opt/pricing_grid.hpp"

#include <cmath>

#include "opt/black_scholes.hpp"

namespace opt {
namespace {

// Vectorised standard normal CDF over an Eigen array using the Abramowitz &
// Stegun 26.2.17 rational approximation (|error| < 7.5e-8 in Phi). It uses only
// abs / + / * / exp, all of which Eigen lowers onto SIMD packets, so the whole
// grid is computed without a single scalar libm call. The exact scalar engine
// (erfc-based) remains the reference; this is the deliberately-fast bulk path.
[[nodiscard]] Eigen::ArrayXd normal_cdf(const Eigen::ArrayXd& x) {
    constexpr double a = 0.2316419;
    constexpr double b1 = 0.319381530;
    constexpr double b2 = -0.356563782;
    constexpr double b3 = 1.781477937;
    constexpr double b4 = -1.821255978;
    constexpr double b5 = 1.330274429;
    constexpr double inv_sqrt_2pi = 0.39894228040143267794;

    const Eigen::ArrayXd abs_x = x.abs();
    const Eigen::ArrayXd t = 1.0 / (1.0 + a * abs_x);
    const Eigen::ArrayXd poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))));
    const Eigen::ArrayXd upper_tail = inv_sqrt_2pi * (-0.5 * abs_x * abs_x).exp() * poly;
    return (x >= 0.0).select(1.0 - upper_tail, upper_tail);
}

}  // namespace

Eigen::MatrixXd price_grid(const MarketData& base, const Eigen::VectorXd& spots,
                           const Eigen::VectorXd& strikes, double expiry, OptionType type) {
    const Eigen::Index num_strikes = strikes.size();
    Eigen::MatrixXd out(spots.size(), num_strikes);

    const double std_dev = base.volatility * std::sqrt(expiry);
    const double disc_r = std::exp(-base.rate * expiry);
    const double disc_q = std::exp(-base.dividend * expiry);
    const double drift = (base.rate - base.dividend) * expiry;
    const double sign = type == OptionType::Call ? 1.0 : -1.0;

    const Eigen::ArrayXd spot = spots.array();
    const Eigen::ArrayXd log_spot = spot.log();
    const Eigen::ArrayXd forward = spot * disc_q;

    // One vectorised pass down the spot axis per strike column.
    for (Eigen::Index j = 0; j < num_strikes; ++j) {
        const double strike = strikes(j);
        const Eigen::ArrayXd d1 =
            (log_spot - std::log(strike) + drift) / std_dev + 0.5 * std_dev;
        const Eigen::ArrayXd d2 = d1 - std_dev;
        const double pv_strike = strike * disc_r;
        // Unified call/put: sign = +1 prices a call, -1 a put.
        out.col(j) = (sign * (forward * normal_cdf(sign * d1) -
                              pv_strike * normal_cdf(sign * d2)))
                         .matrix();
    }
    return out;
}

Eigen::MatrixXd price_grid_scalar(const MarketData& base, const Eigen::VectorXd& spots,
                                  const Eigen::VectorXd& strikes, double expiry, OptionType type) {
    Eigen::MatrixXd out(spots.size(), strikes.size());
    MarketData market = base;
    for (Eigen::Index i = 0; i < spots.size(); ++i) {
        market.spot = spots(i);
        for (Eigen::Index j = 0; j < strikes.size(); ++j) {
            out(i, j) = black_scholes_price(market, strikes(j), expiry, type);
        }
    }
    return out;
}

}  // namespace opt
