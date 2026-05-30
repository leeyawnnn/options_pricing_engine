#include "opt/greeks.hpp"

#include <cmath>

#include "bs_internal.hpp"
#include "opt/black_scholes.hpp"
#include "opt/normal_distribution.hpp"

namespace opt {
namespace {

// ---------------------------------------------------------------------------
// Analytical Greeks expressed in terms of the shared BsTerms, so the bundle
// (bs_greeks) can reuse a single compute_terms() call.
// ---------------------------------------------------------------------------

[[nodiscard]] double delta_from(const detail::BsTerms& t, OptionType type) noexcept {
    // put delta = call delta - e^{-qT} (differentiate put-call parity in S).
    const double call_delta =
        t.disc_q * (t.degenerate ? (t.forward > t.pv_strike ? 1.0 : 0.0) : cumulative_normal(t.d1));
    return type == OptionType::Call ? call_delta : call_delta - t.disc_q;
}

[[nodiscard]] double gamma_from(const MarketData& market, const detail::BsTerms& t) noexcept {
    if (t.degenerate) {
        return 0.0;  // a delta function at the forward; reported as 0.
    }
    return t.disc_q * standard_normal_pdf(t.d1) / (market.spot * t.std_dev);
}

[[nodiscard]] double vega_from(const MarketData& market, double expiry,
                               const detail::BsTerms& t) noexcept {
    if (t.degenerate) {
        return 0.0;
    }
    return market.spot * t.disc_q * standard_normal_pdf(t.d1) * std::sqrt(expiry);
}

[[nodiscard]] double theta_from(const MarketData& market, double expiry, const detail::BsTerms& t,
                                OptionType type) noexcept {
    if (t.degenerate) {
        return 0.0;
    }
    const double decay =
        -(market.spot * t.disc_q * standard_normal_pdf(t.d1) * market.volatility) /
        (2.0 * std::sqrt(expiry));
    if (type == OptionType::Call) {
        return decay - market.rate * t.pv_strike * cumulative_normal(t.d2) +
               market.dividend * t.forward * cumulative_normal(t.d1);
    }
    return decay + market.rate * t.pv_strike * cumulative_normal(-t.d2) -
           market.dividend * t.forward * cumulative_normal(-t.d1);
}

[[nodiscard]] double rho_from(double strike, double expiry, const detail::BsTerms& t,
                              OptionType type) noexcept {
    if (t.degenerate) {
        return 0.0;
    }
    if (type == OptionType::Call) {
        return strike * expiry * t.disc_r * cumulative_normal(t.d2);
    }
    return -strike * expiry * t.disc_r * cumulative_normal(-t.d2);
}

}  // namespace

double bs_delta(const MarketData& market, double strike, double expiry, OptionType type) noexcept {
    return delta_from(detail::compute_terms(market, strike, expiry), type);
}

double bs_gamma(const MarketData& market, double strike, double expiry) noexcept {
    return gamma_from(market, detail::compute_terms(market, strike, expiry));
}

double bs_vega(const MarketData& market, double strike, double expiry) noexcept {
    return vega_from(market, expiry, detail::compute_terms(market, strike, expiry));
}

double bs_theta(const MarketData& market, double strike, double expiry, OptionType type) noexcept {
    return theta_from(market, expiry, detail::compute_terms(market, strike, expiry), type);
}

double bs_rho(const MarketData& market, double strike, double expiry, OptionType type) noexcept {
    return rho_from(strike, expiry, detail::compute_terms(market, strike, expiry), type);
}

Greeks bs_greeks(const MarketData& market, double strike, double expiry, OptionType type) noexcept {
    const detail::BsTerms t = detail::compute_terms(market, strike, expiry);
    return Greeks{delta_from(t, type), gamma_from(market, t), vega_from(market, expiry, t),
                  theta_from(market, expiry, t, type), rho_from(strike, expiry, t, type)};
}

Greeks finite_difference_greeks(const MarketData& market, double strike, double expiry,
                                OptionType type) noexcept {
    // Central-difference step sizes. The near-optimal choice balances O(h^2)
    // truncation against O(eps/h) round-off, and differs by derivative order:
    //   * first derivatives (delta, vega, rho, theta) ~ eps^(1/3) ~ 1e-5;
    //   * second derivative  (gamma)                  ~ eps^(1/4) ~ 1e-4.
    // The spot bump is relative (S is scale-dependent); vol/rate/time already
    // live near O(1), so an absolute bump is fine.
    const double h_spot = market.spot * 1e-4;  // shared by delta and gamma
    constexpr double h_vol = 1e-5;
    constexpr double h_rate = 1e-5;
    constexpr double h_time = 1e-5;

    const auto price = [&](const MarketData& m, double t) noexcept {
        return black_scholes_price(m, strike, t, type);
    };

    MarketData up = market;
    MarketData down = market;

    up.spot = market.spot + h_spot;
    down.spot = market.spot - h_spot;
    const double price_up = price(up, expiry);
    const double price_down = price(down, expiry);
    const double price_mid = price(market, expiry);
    const double delta = (price_up - price_down) / (2.0 * h_spot);
    const double gamma = (price_up - 2.0 * price_mid + price_down) / (h_spot * h_spot);

    up = market;
    down = market;
    up.volatility = market.volatility + h_vol;
    down.volatility = market.volatility - h_vol;
    const double vega = (price(up, expiry) - price(down, expiry)) / (2.0 * h_vol);

    up = market;
    down = market;
    up.rate = market.rate + h_rate;
    down.rate = market.rate - h_rate;
    const double rho = (price(up, expiry) - price(down, expiry)) / (2.0 * h_rate);

    // theta = dV/dt(calendar) = -dV/dT(time-to-expiry).
    const double theta =
        -(price(market, expiry + h_time) - price(market, expiry - h_time)) / (2.0 * h_time);

    return Greeks{delta, gamma, vega, theta, rho};
}

}  // namespace opt
