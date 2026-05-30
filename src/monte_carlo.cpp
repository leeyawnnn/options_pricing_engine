#include "opt/monte_carlo.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

namespace opt {
namespace {

// 97.5th percentile of the standard normal, for a two-sided 95% interval.
constexpr double kZ975 = 1.959963984540054;

struct SampleStats {
    double mean;
    double variance;  // unbiased (divided by n - 1)
};

[[nodiscard]] SampleStats mean_and_variance(const std::vector<double>& values) noexcept {
    const auto n = static_cast<double>(values.size());
    double sum = 0.0;
    for (const double v : values) {
        sum += v;
    }
    const double mean = sum / n;
    double sq = 0.0;
    for (const double v : values) {
        const double d = v - mean;
        sq += d * d;
    }
    return {mean, sq / (n - 1.0)};
}

}  // namespace

McResult monte_carlo_price(const Option& option, const MarketData& market,
                           const McSettings& settings) {
    const bool antithetic = settings.variance_reduction == VarianceReduction::Antithetic ||
                            settings.variance_reduction == VarianceReduction::AntitheticAndControl;
    const bool control = settings.variance_reduction == VarianceReduction::ControlVariate ||
                         settings.variance_reduction == VarianceReduction::AntitheticAndControl;

    // Antithetic sampling spends two evaluations per independent sample.
    const std::size_t samples = antithetic ? settings.num_paths / 2 : settings.num_paths;
    if (samples < 2) {
        throw std::invalid_argument("monte_carlo_price: need at least 2 samples");
    }

    const double expiry = option.expiry();
    const double vol = market.volatility;
    const double drift = (market.rate - market.dividend - 0.5 * vol * vol) * expiry;
    const double vol_sqrt_t = vol * std::sqrt(expiry);
    const double discount = std::exp(-market.rate * expiry);
    // Known mean of the control E[e^{-rT} S_T] = S_0 e^{-qT} (the BSM forward,
    // discounted) -- the analytically known anchor the control variate needs.
    const double control_mean = market.spot * std::exp(-market.dividend * expiry);

    std::mt19937_64 rng{settings.seed};
    std::normal_distribution<double> standard_normal{0.0, 1.0};

    const auto terminal = [&](double z) noexcept {
        return market.spot * std::exp(drift + vol_sqrt_t * z);
    };

    std::vector<double> payoffs(samples);
    std::vector<double> controls(control ? samples : 0);

    for (std::size_t i = 0; i < samples; ++i) {
        const double z = standard_normal(rng);
        const double st = terminal(z);
        double payoff = discount * option.payoff(st);
        double ctrl = discount * st;
        if (antithetic) {
            const double st_mirror = terminal(-z);
            payoff = 0.5 * (payoff + discount * option.payoff(st_mirror));
            ctrl = 0.5 * (ctrl + discount * st_mirror);
        }
        payoffs[i] = payoff;
        if (control) {
            controls[i] = ctrl;
        }
    }

    const SampleStats px = mean_and_variance(payoffs);
    double price = px.mean;
    double estimator_variance = px.variance;

    if (control) {
        const SampleStats cx = mean_and_variance(controls);
        double cov_acc = 0.0;
        for (std::size_t i = 0; i < samples; ++i) {
            cov_acc += (payoffs[i] - px.mean) * (controls[i] - cx.mean);
        }
        const double covariance = cov_acc / (static_cast<double>(samples) - 1.0);
        // Optimal coefficient beta* = Cov(X, C) / Var(C); the residual variance
        // is Var(X) - Cov^2/Var(C) = Var(X) - beta * Cov.
        const double beta = cx.variance > 0.0 ? covariance / cx.variance : 0.0;
        price = px.mean - beta * (cx.mean - control_mean);
        estimator_variance = std::max(px.variance - beta * covariance, 0.0);
    }

    const double std_error = std::sqrt(estimator_variance / static_cast<double>(samples));
    return McResult{price, std_error, price - kZ975 * std_error, price + kZ975 * std_error, samples};
}

}  // namespace opt
