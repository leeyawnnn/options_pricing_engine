#pragma once

#include <cstddef>
#include <cstdint>

#include "opt/market_data.hpp"
#include "opt/option.hpp"

namespace opt {

/// Variance-reduction technique applied to the Monte Carlo estimator.
enum class VarianceReduction {
    None,                  ///< Plain Monte Carlo.
    Antithetic,            ///< Pair each draw Z with its mirror -Z.
    ControlVariate,        ///< Use the discounted terminal spot as a control.
    AntitheticAndControl,  ///< Both of the above together.
};

/// Knobs for a Monte Carlo pricing run. Defaults are deterministic so test
/// results are reproducible.
struct McSettings {
    std::size_t num_paths = 100'000;        ///< Terminal-price evaluations.
    std::uint64_t seed = 0x5DEECE66DULL;    ///< Seed for the mt19937_64 engine.
    VarianceReduction variance_reduction = VarianceReduction::None;
};

/// Outcome of a Monte Carlo pricing run, including a 95% confidence interval.
struct McResult {
    double price;          ///< Point estimate of the option value.
    double std_error;      ///< Standard error of the estimate.
    double ci_low;         ///< Lower bound of the 95% confidence interval.
    double ci_high;        ///< Upper bound of the 95% confidence interval.
    std::size_t samples;   ///< Number of independent samples averaged.
};

/// Price a European-style option by simulating terminal underlying prices under
/// geometric Brownian motion, \f$S_T = S_0\exp\!\big((r-q-\tfrac12\sigma^2)T +
/// \sigma\sqrt{T}\,Z\big)\f$, and averaging discounted payoffs.
///
/// Exercise is treated as European: only the terminal price matters, so an
/// American option passed here is valued on its terminal payoff alone.
///
/// @throws std::invalid_argument if the run would have fewer than 2 samples.
[[nodiscard]] McResult monte_carlo_price(const Option& option, const MarketData& market,
                                         const McSettings& settings);

}  // namespace opt
