#pragma once

#include "opt/market_data.hpp"
#include "opt/option.hpp"

namespace opt {

/// First- and second-order Black-Scholes sensitivities ("the Greeks").
///
/// Units follow the most common market conventions:
///   - @ref delta, @ref gamma are dimensionless / per unit underlying.
///   - @ref vega is per **unit** volatility (a move of 1.00 = 100 vol points);
///     divide by 100 for the per-1%-point figure traders usually quote.
///   - @ref theta is per **year** and is \f$\partial V/\partial t\f$ (calendar
///     time), so it is negative for a long option; divide by 365 for per-day.
///   - @ref rho is per **unit** interest rate (a move of 1.00 = 100 bp × 100);
///     divide by 100 for the per-1%-point figure.
struct Greeks {
    double delta;  ///< \f$\partial V/\partial S\f$
    double gamma;  ///< \f$\partial^2 V/\partial S^2\f$
    double vega;   ///< \f$\partial V/\partial \sigma\f$
    double theta;  ///< \f$\partial V/\partial t\f$ (per year)
    double rho;    ///< \f$\partial V/\partial r\f$
};

/// Analytical Black-Scholes delta, \f$e^{-qT}\Phi(d_1)\f$ for a call and
/// \f$e^{-qT}(\Phi(d_1)-1)\f$ for a put.
[[nodiscard]] double bs_delta(const MarketData& market, double strike, double expiry,
                              OptionType type) noexcept;

/// Analytical gamma, \f$\dfrac{e^{-qT}\phi(d_1)}{S\sigma\sqrt{T}}\f$. Identical
/// for calls and puts, hence no OptionType parameter.
[[nodiscard]] double bs_gamma(const MarketData& market, double strike, double expiry) noexcept;

/// Analytical vega, \f$S e^{-qT}\phi(d_1)\sqrt{T}\f$ (per unit volatility).
/// Identical for calls and puts.
[[nodiscard]] double bs_vega(const MarketData& market, double strike, double expiry) noexcept;

/// Analytical theta (per year), \f$\partial V/\partial t\f$.
[[nodiscard]] double bs_theta(const MarketData& market, double strike, double expiry,
                              OptionType type) noexcept;

/// Analytical rho (per unit rate), \f$KTe^{-rT}\Phi(d_2)\f$ for a call and
/// \f$-KTe^{-rT}\Phi(-d_2)\f$ for a put.
[[nodiscard]] double bs_rho(const MarketData& market, double strike, double expiry,
                            OptionType type) noexcept;

/// Analytical vanna, \f$\partial^2 V/\partial S\,\partial\sigma =
/// -e^{-qT}\phi(d_1)\,d_2/\sigma\f$.
///
/// Vanna is the same object read two ways: the sensitivity of delta to a change
/// in volatility, and the sensitivity of vega to a change in spot. It is what a
/// vol desk quotes as the risk-reversal exposure of a position -- it is the
/// Greek that tells you how a skew move re-hedges you -- and it changes sign
/// through the money, so a book that is vanna-flat at one spot is not at
/// another. Identical for calls and puts, hence no OptionType parameter.
///
/// Units follow @ref Greeks::vega: per unit volatility.
[[nodiscard]] double bs_vanna(const MarketData& market, double strike, double expiry) noexcept;

/// Analytical volga (also called vomma), \f$\partial^2 V/\partial\sigma^2 =
/// \mathrm{vega}\cdot d_1 d_2/\sigma\f$.
///
/// Volga is the convexity of the position in volatility, and it is why a
/// vega-neutral book still makes or loses money when vol moves: vega itself is
/// not constant. It is smallest at the money (where \f$d_1 d_2 < 0\f$, so volga
/// is actually negative for a near-ATM option) and largest in the wings, which
/// is the analytical reason a butterfly is a long-vol-of-vol position.
///
/// Units: per unit volatility squared. Identical for calls and puts.
[[nodiscard]] double bs_volga(const MarketData& market, double strike, double expiry) noexcept;

/// All five analytical Greeks, sharing a single evaluation of the
/// Black-Scholes intermediate terms.
[[nodiscard]] Greeks bs_greeks(const MarketData& market, double strike, double expiry,
                               OptionType type) noexcept;

/// Greeks computed by central finite differences against the analytical
/// Black-Scholes price. Model-agnostic and useful as an independent check on
/// (or stand-in for) the closed-form formulas.
[[nodiscard]] Greeks finite_difference_greeks(const MarketData& market, double strike,
                                              double expiry, OptionType type) noexcept;

}  // namespace opt
