// Unit tests for the analytical and finite-difference Black-Scholes Greeks.
#include "opt/greeks.hpp"

#include <gtest/gtest.h>

#include <cmath>

#include "opt/market_data.hpp"
#include "opt/option.hpp"

namespace {

// Hand-checked analytical values for S=100, K=100, r=0.05, q=0, sigma=0.2,
// T=1 (so d1=0.35, d2=0.15).
TEST(Greeks, AnalyticalKnownValues) {
    const opt::MarketData mkt{100.0, 0.05, 0.0, 0.20};
    const opt::Greeks g = opt::bs_greeks(mkt, 100.0, 1.0, opt::OptionType::Call);
    EXPECT_NEAR(g.delta, 0.636831, 1e-4);
    EXPECT_NEAR(g.gamma, 0.018762, 1e-4);
    EXPECT_NEAR(g.vega, 37.52403, 1e-3);
    EXPECT_NEAR(g.theta, -6.41403, 1e-3);
    EXPECT_NEAR(g.rho, 53.23248, 1e-3);
}

// Analytical Greeks must match central finite differences across a sweep.
// The erfc-based CDF makes the price exact to ~1e-16, so the only error here is
// finite-difference truncation/round-off, which the step sizes keep well below
// the 1e-4 bound for every Greek across the grid.
TEST(Greeks, AnalyticalMatchesFiniteDifference) {
    const double spots[] = {80.0, 100.0, 120.0};
    const double strikes[] = {90.0, 100.0, 110.0};
    const double rates[] = {0.01, 0.05};
    const double dividends[] = {0.0, 0.03};
    const double vols[] = {0.15, 0.30};
    const double expiries[] = {0.25, 1.0, 2.0};

    for (const double s : spots) {
        for (const double k : strikes) {
            for (const double r : rates) {
                for (const double q : dividends) {
                    for (const double vol : vols) {
                        for (const double t : expiries) {
                            const opt::MarketData mkt{s, r, q, vol};
                            for (const opt::OptionType type :
                                 {opt::OptionType::Call, opt::OptionType::Put}) {
                                const opt::Greeks a = opt::bs_greeks(mkt, k, t, type);
                                const opt::Greeks fd =
                                    opt::finite_difference_greeks(mkt, k, t, type);
                                SCOPED_TRACE(::testing::Message()
                                             << "S=" << s << " K=" << k << " r=" << r << " q=" << q
                                             << " vol=" << vol << " T=" << t << " type="
                                             << (type == opt::OptionType::Call ? "call" : "put"));
                                EXPECT_NEAR(a.delta, fd.delta, 1e-4);
                                EXPECT_NEAR(a.gamma, fd.gamma, 1e-4);
                                EXPECT_NEAR(a.vega, fd.vega, 1e-4);
                                EXPECT_NEAR(a.theta, fd.theta, 1e-4);
                                EXPECT_NEAR(a.rho, fd.rho, 1e-4);
                            }
                        }
                    }
                }
            }
        }
    }
}

// Without dividends, |call delta| + |put delta| = 1.
TEST(Greeks, DeltaParityNoDividends) {
    const opt::MarketData mkt{100.0, 0.05, 0.0, 0.25};
    for (const double k : {80.0, 100.0, 120.0}) {
        const double call_delta = opt::bs_delta(mkt, k, 1.0, opt::OptionType::Call);
        const double put_delta = opt::bs_delta(mkt, k, 1.0, opt::OptionType::Put);
        EXPECT_NEAR(call_delta + std::abs(put_delta), 1.0, 1e-12);
    }
}

// With dividends the general identity is call delta - put delta = e^{-qT}.
TEST(Greeks, DeltaDifferenceEqualsDividendDiscount) {
    const opt::MarketData mkt{100.0, 0.05, 0.04, 0.25};
    const double expiry = 1.5;
    const double call_delta = opt::bs_delta(mkt, 105.0, expiry, opt::OptionType::Call);
    const double put_delta = opt::bs_delta(mkt, 105.0, expiry, opt::OptionType::Put);
    EXPECT_NEAR(call_delta - put_delta, std::exp(-mkt.dividend * expiry), 1e-12);
}

// Gamma and vega do not depend on call/put.
TEST(Greeks, GammaAndVegaIdenticalForCallAndPut) {
    const opt::MarketData mkt{100.0, 0.05, 0.02, 0.30};
    const opt::Greeks call = opt::bs_greeks(mkt, 110.0, 0.75, opt::OptionType::Call);
    const opt::Greeks put = opt::bs_greeks(mkt, 110.0, 0.75, opt::OptionType::Put);
    EXPECT_DOUBLE_EQ(call.gamma, put.gamma);
    EXPECT_DOUBLE_EQ(call.vega, put.vega);
}

}  // namespace
