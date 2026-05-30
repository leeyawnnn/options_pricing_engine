#pragma once

#include <Eigen/Core>

#include "opt/market_data.hpp"
#include "opt/option.hpp"

namespace opt {

/// Price a grid of European options under Black-Scholes for every
/// (spot, strike) pair at a single expiry, vectorised with Eigen.
///
/// The result is an `spots.size() x strikes.size()` matrix whose entry
/// `(i, j)` is the price for spot `spots(i)` and strike `strikes(j)`. The
/// market's rate, dividend and volatility come from @p base; its `spot` field
/// is ignored (the spot axis is swept by @p spots).
///
/// Precondition: every spot and strike is positive, volatility > 0 and
/// @p expiry > 0 (the vectorised path does not special-case the degenerate
/// zero-variance limit).
[[nodiscard]] Eigen::MatrixXd price_grid(const MarketData& base, const Eigen::VectorXd& spots,
                                         const Eigen::VectorXd& strikes, double expiry,
                                         OptionType type);

/// Same result as price_grid(), computed with a plain scalar double loop over
/// the engine's black_scholes_price(). Kept for validation and for the
/// loop-vs-vectorised benchmark comparison.
[[nodiscard]] Eigen::MatrixXd price_grid_scalar(const MarketData& base,
                                                const Eigen::VectorXd& spots,
                                                const Eigen::VectorXd& strikes, double expiry,
                                                OptionType type);

}  // namespace opt
