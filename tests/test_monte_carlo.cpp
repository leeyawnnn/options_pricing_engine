// Unit tests for the Monte Carlo pricer and its variance-reduction variants.
#include "opt/monte_carlo.hpp"

#include <gtest/gtest.h>

#include <cmath>

#include "opt/black_scholes.hpp"
#include "opt/market_data.hpp"
#include "opt/option.hpp"

namespace {

const opt::MarketData kMkt{100.0, 0.05, 0.0, 0.20};
constexpr double kZ975 = 1.959963984540054;

opt::McResult price(const opt::Option& option, opt::VarianceReduction vr, std::size_t paths,
                    std::uint64_t seed = 20240530ULL) {
    opt::McSettings settings;
    settings.num_paths = paths;
    settings.seed = seed;
    settings.variance_reduction = vr;
    return opt::monte_carlo_price(option, kMkt, settings);
}

TEST(MonteCarlo, ConvergesToBlackScholesWithinThreeStdErrors) {
    const opt::EuropeanCall call{100.0, 1.0};
    const double bs = opt::black_scholes_call(kMkt, 100.0, 1.0);
    const opt::McResult r = price(call, opt::VarianceReduction::None, 100'000);
    EXPECT_LT(std::abs(r.price - bs), 3.0 * r.std_error);
    EXPECT_EQ(r.samples, 100'000u);
}

TEST(MonteCarlo, ConfidenceIntervalIsCenteredAndCovers) {
    const opt::EuropeanCall call{100.0, 1.0};
    const double bs = opt::black_scholes_call(kMkt, 100.0, 1.0);
    const opt::McResult r = price(call, opt::VarianceReduction::ControlVariate, 100'000);
    EXPECT_NEAR(r.ci_low, r.price - kZ975 * r.std_error, 1e-12);
    EXPECT_NEAR(r.ci_high, r.price + kZ975 * r.std_error, 1e-12);
    EXPECT_LT(r.ci_low, bs);
    EXPECT_GT(r.ci_high, bs);
}

TEST(MonteCarlo, DeterministicForFixedSeed) {
    const opt::EuropeanCall call{100.0, 1.0};
    const opt::McResult a = price(call, opt::VarianceReduction::Antithetic, 50'000, 777ULL);
    const opt::McResult b = price(call, opt::VarianceReduction::Antithetic, 50'000, 777ULL);
    EXPECT_DOUBLE_EQ(a.price, b.price);
    EXPECT_DOUBLE_EQ(a.std_error, b.std_error);
}

TEST(MonteCarlo, VarianceReductionLowersStandardError) {
    const opt::EuropeanCall call{100.0, 1.0};
    const std::size_t paths = 200'000;
    const double plain = price(call, opt::VarianceReduction::None, paths).std_error;
    const double antithetic = price(call, opt::VarianceReduction::Antithetic, paths).std_error;
    const double control = price(call, opt::VarianceReduction::ControlVariate, paths).std_error;
    const double both = price(call, opt::VarianceReduction::AntitheticAndControl, paths).std_error;

    EXPECT_LT(antithetic, plain);
    EXPECT_LT(control, plain);
    EXPECT_LT(both, plain);
}

TEST(MonteCarlo, PutConvergesWithVarianceReduction) {
    const opt::EuropeanPut put{100.0, 1.0};
    const double bs = opt::black_scholes_put(kMkt, 100.0, 1.0);
    const opt::McResult r = price(put, opt::VarianceReduction::AntitheticAndControl, 100'000);
    EXPECT_LT(std::abs(r.price - bs), 3.0 * r.std_error);
}

TEST(MonteCarlo, RejectsTooFewSamples) {
    const opt::EuropeanCall call{100.0, 1.0};
    EXPECT_THROW((void)price(call, opt::VarianceReduction::None, 1), std::invalid_argument);
    // Antithetic halves the sample count, so 2 paths -> 1 sample -> rejected.
    EXPECT_THROW((void)price(call, opt::VarianceReduction::Antithetic, 2), std::invalid_argument);
}

}  // namespace
