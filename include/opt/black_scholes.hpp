#pragma once

#include "opt/market_data.hpp"
#include "opt/option.hpp"

namespace opt {

/// Black-Scholes-Merton price of a European call.
///
/// \f$C = S e^{-qT} \Phi(d_1) - K e^{-rT} \Phi(d_2)\f$, with
/// \f$d_{1,2} = \frac{\ln(S/K) + (r - q \pm \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}\f$.
///
/// In the degenerate limit \f$\sigma\sqrt{T} \to 0\f$ (zero volatility or zero
/// time) the price collapses to the present value of the certain payoff,
/// \f$\max(S e^{-qT} - K e^{-rT}, 0)\f$.
///
/// @param market Market observables (spot, rate, dividend, volatility).
/// @param strike Strike price \f$K > 0\f$.
/// @param expiry Time to expiry in years \f$T \ge 0\f$.
[[nodiscard]] double black_scholes_call(const MarketData& market, double strike,
                                        double expiry) noexcept;

/// Black-Scholes-Merton price of a European put,
/// \f$P = K e^{-rT} \Phi(-d_2) - S e^{-qT} \Phi(-d_1)\f$.
/// @see black_scholes_call for parameter semantics.
[[nodiscard]] double black_scholes_put(const MarketData& market, double strike,
                                       double expiry) noexcept;

/// Dispatches to black_scholes_call or black_scholes_put on @p type.
[[nodiscard]] double black_scholes_price(const MarketData& market, double strike, double expiry,
                                         OptionType type) noexcept;

/// Put-call parity residual: \f$(C - P) - \bigl(S e^{-qT} - K e^{-rT}\bigr)\f$.
/// This is identically zero in exact arithmetic, so the returned value measures
/// only floating-point error and is a useful internal consistency check.
[[nodiscard]] double put_call_parity_residual(const MarketData& market, double strike,
                                              double expiry) noexcept;

}  // namespace opt
