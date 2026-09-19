#include "opt/binomial_tree.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace opt {
namespace {

// Zero volatility collapses the lattice: u = e^{0} = 1, d = 1/u = 1, and the
// risk-neutral probability (growth - d)/(u - d) is 0/0. The CRR recursion
// cannot be evaluated there, so the degenerate case is priced directly.
//
// With sigma = 0 the underlying is deterministic, S_t = S_0 e^{(r-q)t}, so a
// European option is worth the discounted payoff of that single path. An
// American option is worth the best discounted payoff available on the exercise
// grid, which is not the same thing: a deep in-the-money put is worth exercising
// immediately rather than waiting for the forward to drift further away.
//
// The exercise grid is deliberately the same n*dt grid the stochastic path
// uses, so the two branches agree in the sigma -> 0 limit instead of jumping.
[[nodiscard]] double deterministic_price(const Option& option, const MarketData& market,
                                         int steps) noexcept {
    const double expiry = option.expiry();
    const double drift = market.rate - market.dividend;
    const double dt = expiry / steps;

    double value = 0.0;
    for (int step = steps; step >= 0; --step) {
        const double time = dt * step;
        const double spot = market.spot * std::exp(drift * time);
        const double exercise = std::exp(-market.rate * time) * option.payoff(spot);
        if (step == steps) {
            value = exercise;
        } else if (option.is_american()) {
            value = std::max(value, exercise);
        }
    }
    // `value` is a present value already (every term carries e^{-r t}).
    return value;
}

// Smallest step count at which a CRR lattice for these inputs still has a
// risk-neutral probability in [0, 1]: sigma >= |r-q| sqrt(T/N) rearranges to
// N >= (r-q)^2 T / sigma^2. Returned as a string because its only use is an
// error message, and because for a volatility near zero the exact value
// overflows anything an int could hold.
[[nodiscard]] std::string minimum_steps_for(const MarketData& market, double expiry) {
    const double drift = std::abs(market.rate - market.dividend);
    const double required = drift * drift * expiry / (market.volatility * market.volatility);
    if (!std::isfinite(required) || required > 1e15) {
        return "an impractical number of";
    }
    return "at least " + std::to_string(static_cast<long long>(std::ceil(required)) + 1);
}

}  // namespace

double binomial_price(const Option& option, const MarketData& market, int steps) {
    if (steps < 1) {
        throw std::invalid_argument("binomial_price: steps must be >= 1");
    }

    const double expiry = option.expiry();
    const double spot = market.spot;
    if (expiry <= 0.0) {
        return option.payoff(spot);  // already expired: value is the intrinsic payoff
    }
    if (!(market.volatility > 0.0)) {
        return deterministic_price(option, market, steps);
    }

    const double dt = expiry / steps;
    const double up = std::exp(market.volatility * std::sqrt(dt));
    const double down = 1.0 / up;
    const double growth = std::exp((market.rate - market.dividend) * dt);
    const double discount = std::exp(-market.rate * dt);
    const double p_up = (growth - down) / (up - down);
    // The lattice is only a lattice while p is a probability. p leaves [0, 1]
    // exactly when the per-step drift outruns the per-step move, |r-q| dt >
    // sigma sqrt(dt), and the rollback below silently stops being an
    // expectation: with a negative weight on one branch it returned 0 for an
    // at-the-forward call worth 0.035. A refused answer beats a wrong one, and
    // since the floor is |r-q| sqrt(T/N), more steps is the cure.
    if (!std::isfinite(p_up) || p_up < 0.0 || p_up > 1.0) {
        throw std::invalid_argument(
            "binomial_price: volatility " + std::to_string(market.volatility) +
            " is below the CRR stability floor " +
            std::to_string(std::abs(market.rate - market.dividend) * std::sqrt(dt)) + " for " +
            std::to_string(steps) + " steps, so the risk-neutral probability (" +
            std::to_string(p_up) + ") is not in [0, 1]; use " + minimum_steps_for(market, expiry) +
            " steps");
    }
    const double p_down = 1.0 - p_up;
    const bool american = option.is_american();

    const auto n = static_cast<std::size_t>(steps);

    // Every node in the tree sits at S * e^{k * sigma sqrt(dt)} for an integer k
    // in [-N, N], so the whole lattice of spots is this one ladder, indexed by
    // k + N. Node j at step i has k = 2j - i.
    //
    // Building it from the exponent matters. The obvious alternative -- start at
    // S * d^N and multiply up by u/d -- underflows to exactly zero once
    // sigma sqrt(dt) * N exceeds ~709, and zero times anything is zero, so the
    // entire terminal layer collapses to a spot of zero and the tree prices a
    // worthless option. That is reachable: at 25600 steps and 500% vol it
    // returned 0.00 for a call worth 97.77. Here only the genuinely negligible
    // tail entries underflow, which is the right answer for them.
    const double log_step = market.volatility * std::sqrt(dt);

    // The ladder spans S * e^{+/- sigma sqrt(dt) N}, so the lattice only exists
    // as doubles while that exponent stays inside exp's range. Past it the top
    // of the ladder is +inf and the bottom is 0, and the rollback launders both
    // into a confident-looking price: an inf terminal payoff propagates all the
    // way down. Since sigma sqrt(dt) N = sigma sqrt(T) sqrt(N), this only binds
    // for extreme volatilities, which is exactly where a solver hunting for a
    // bracket ends up.
    constexpr double max_log_span = 700.0;  // exp overflows just past 709
    if (log_step * steps > max_log_span) {
        throw std::invalid_argument(
            "binomial_price: volatility " + std::to_string(market.volatility) + " over " +
            std::to_string(steps) + " steps spans e^" + std::to_string(log_step * steps) +
            " in spot, beyond the range of a double; reduce the volatility or the step count");
    }

    std::vector<double> spot_ladder(2 * n + 1);
    for (std::size_t k = 0; k <= 2 * n; ++k) {
        spot_ladder[k] =
            spot * std::exp((static_cast<double>(k) - static_cast<double>(n)) * log_step);
    }

    // Terminal layer: node j has k = 2j - N, i.e. ladder index 2j.
    std::vector<double> values(n + 1);
    for (std::size_t j = 0; j <= n; ++j) {
        values[j] = option.payoff(spot_ladder[2 * j]);
    }

    // Roll back through the tree, overwriting `values` in place. At step i,
    // node j discounts the expectation of its two children (indices j and j+1
    // in the current buffer); ascending j is safe because values[j] is read
    // before it is overwritten and values[j+1] is not yet touched.
    for (int i = steps - 1; i >= 0; --i) {
        if (american) {
            // Node j at step i has k = 2j - i, i.e. ladder index (N - i) + 2j.
            const std::size_t ladder_base = n - static_cast<std::size_t>(i);
            for (int j = 0; j <= i; ++j) {
                const auto idx = static_cast<std::size_t>(j);
                const double continuation =
                    discount * (p_up * values[idx + 1] + p_down * values[idx]);
                const double node_spot = spot_ladder[ladder_base + 2 * idx];
                values[idx] = std::max(continuation, option.payoff(node_spot));
            }
        } else {
            for (int j = 0; j <= i; ++j) {
                const auto idx = static_cast<std::size_t>(j);
                values[idx] = discount * (p_up * values[idx + 1] + p_down * values[idx]);
            }
        }
    }

    return values[0];
}

}  // namespace opt
