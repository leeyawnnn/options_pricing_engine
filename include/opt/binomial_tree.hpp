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
/// A CRR lattice is only a valid discretisation while \f$p \in [0,1]\f$, which
/// holds exactly when the per-step move covers the per-step drift:
/// \f$\sigma\sqrt{\Delta t} \ge |r-q|\,\Delta t\f$, i.e.
/// \f$\sigma \ge |r-q|\sqrt{T/N}\f$. Below that floor \f$p\f$ leaves
/// \f$[0,1]\f$, the rollback stops being an expectation, and the number it
/// produces is meaningless rather than merely imprecise -- so this function
/// rejects that regime instead of returning it. The floor falls as \f$N\f$
/// grows, so the cure is more steps, and the exception says how many.
///
/// Accuracy degrades continuously as that floor is approached, well before it is
/// crossed: at \f$p \approx 0.95\f$ the tree is so one-sided that it is several
/// tens of percent below the analytic value. Treat the floor as the point where
/// a wrong answer becomes a refused one, not as the edge of the trustworthy
/// region.
///
/// @param option Contract terms (strike, expiry, call/put, exercise style).
/// @param market Market observables (spot, rate, dividend, volatility).
/// @param steps  Number of time steps \f$N \ge 1\f$.
/// @return The present value of the option.
/// @throws std::invalid_argument if @p steps < 1, or if the volatility is below
///         the stability floor \f$|r-q|\sqrt{T/N}\f$ described above.
[[nodiscard]] double binomial_price(const Option& option, const MarketData& market, int steps);

}  // namespace opt
