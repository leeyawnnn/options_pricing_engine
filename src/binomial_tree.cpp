#include "opt/binomial_tree.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace opt {

double binomial_price(const Option& option, const MarketData& market, int steps) {
    if (steps < 1) {
        throw std::invalid_argument("binomial_price: steps must be >= 1");
    }

    const double expiry = option.expiry();
    const double spot = market.spot;
    if (expiry <= 0.0) {
        return option.payoff(spot);  // already expired: value is the intrinsic payoff
    }

    const double dt = expiry / steps;
    const double up = std::exp(market.volatility * std::sqrt(dt));
    const double down = 1.0 / up;
    const double growth = std::exp((market.rate - market.dividend) * dt);
    const double discount = std::exp(-market.rate * dt);
    const double p_up = (growth - down) / (up - down);
    const double p_down = 1.0 - p_up;
    const double up_over_down = up * up;  // u/d = u^2
    const bool american = option.is_american();

    const auto n = static_cast<std::size_t>(steps);

    // Terminal layer: node j (j up-moves) has spot S * u^j * d^(N-j)
    //               = S * d^N * (u/d)^j, built incrementally.
    std::vector<double> values(n + 1);
    double terminal_spot = spot * std::pow(down, steps);
    for (std::size_t j = 0; j <= n; ++j) {
        values[j] = option.payoff(terminal_spot);
        terminal_spot *= up_over_down;
    }

    // Roll back through the tree, overwriting `values` in place. At step i,
    // node j discounts the expectation of its two children (indices j and j+1
    // in the current buffer); ascending j is safe because values[j] is read
    // before it is overwritten and values[j+1] is not yet touched.
    for (int i = steps - 1; i >= 0; --i) {
        if (american) {
            double node_spot = spot * std::pow(down, i);
            for (int j = 0; j <= i; ++j) {
                const auto idx = static_cast<std::size_t>(j);
                const double continuation =
                    discount * (p_up * values[idx + 1] + p_down * values[idx]);
                values[idx] = std::max(continuation, option.payoff(node_spot));
                node_spot *= up_over_down;
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
