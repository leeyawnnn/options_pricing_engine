#pragma once

#include "opt/market_data.hpp"
#include "opt/option.hpp"

namespace opt {

/// Price an option on a Cox-Ross-Rubinstein (CRR) binomial lattice.
///
/// The CRR parameterisation uses \f$u = e^{\sigma\sqrt{\Delta t}}\f$,
/// \f$d = 1/u\f$, and the risk-neutral up-probability
/// \f$p = \dfrac{e^{(r-q)\Delta t} - d}{u - d}\f$ with \f$\Delta t = T/N\f$.
///
/// Exercise style is taken from the option itself: a European option is valued
/// purely by discounted expectation, while an American option additionally
/// compares the continuation value against immediate exercise at every node.
///
/// The implementation keeps only a single price layer (one
/// `std::vector<double>` of size \f$N+1\f$) and rolls it back through the tree,
/// so memory is \f$O(N)\f$ rather than \f$O(N^2)\f$.
///
/// @param option Contract terms (strike, expiry, call/put, exercise style).
/// @param market Market observables (spot, rate, dividend, volatility).
/// @param steps  Number of time steps \f$N \ge 1\f$.
/// @return The present value of the option.
/// @throws std::invalid_argument if @p steps < 1.
[[nodiscard]] double binomial_price(const Option& option, const MarketData& market, int steps);

}  // namespace opt
