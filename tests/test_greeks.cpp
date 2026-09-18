// Unit tests for the analytical and finite-difference Black-Scholes Greeks.
#include <gtest/gtest.h>

#include <cmath>

#include "opt/greeks.hpp"
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

// -----------------------------------------------------------------------------
// Second-order cross Greeks
// -----------------------------------------------------------------------------

namespace {

// Central finite difference of an arbitrary Greek in one market variable.
template <typename Fn>
[[nodiscard]] double central_difference(const opt::MarketData& market, double step,
                                        double opt::MarketData::*field, const Fn& greek) {
    opt::MarketData up = market;
    opt::MarketData down = market;
    up.*field = market.*field + step;
    down.*field = market.*field - step;
    return (greek(up) - greek(down)) / (2.0 * step);
}

}  // namespace

// Vanna is d(delta)/d(sigma) and d(vega)/d(spot) at once. Checking it against
// both finite differences, rather than one, is what makes the test meaningful:
// a sign error would pass one and fail the other.
TEST(Greeks, VannaMatchesBothOfItsFiniteDifferences) {
    for (const double spot : {70.0, 100.0, 130.0}) {
        for (const double vol : {0.12, 0.25, 0.60}) {
            for (const double expiry : {0.25, 1.0, 2.5}) {
                const opt::MarketData market{spot, 0.04, 0.02, vol};
                const double analytic = opt::bs_vanna(market, 100.0, expiry);

                for (const opt::OptionType type : {opt::OptionType::Call, opt::OptionType::Put}) {
                    const double d_delta_d_vol = central_difference(
                        market, 1e-5, &opt::MarketData::volatility, [&](const opt::MarketData& m) {
                            return opt::bs_delta(m, 100.0, expiry, type);
                        });
                    EXPECT_NEAR(analytic, d_delta_d_vol, 1e-6)
                        << "S=" << spot << " vol=" << vol << " T=" << expiry;
                }

                const double d_vega_d_spot = central_difference(
                    market, spot * 1e-5, &opt::MarketData::spot,
                    [&](const opt::MarketData& m) { return opt::bs_vega(m, 100.0, expiry); });
                EXPECT_NEAR(analytic, d_vega_d_spot, 1e-6)
                    << "S=" << spot << " vol=" << vol << " T=" << expiry;
            }
        }
    }
}

TEST(Greeks, VolgaMatchesFiniteDifferenceOfVega) {
    for (const double spot : {70.0, 100.0, 130.0}) {
        for (const double vol : {0.12, 0.25, 0.60}) {
            for (const double expiry : {0.25, 1.0, 2.5}) {
                const opt::MarketData market{spot, 0.04, 0.02, vol};
                const double analytic = opt::bs_volga(market, 100.0, expiry);
                const double numeric = central_difference(
                    market, 1e-5, &opt::MarketData::volatility,
                    [&](const opt::MarketData& m) { return opt::bs_vega(m, 100.0, expiry); });
                EXPECT_NEAR(analytic, numeric, 1e-4 * std::max(std::abs(analytic), 1.0))
                    << "S=" << spot << " vol=" << vol << " T=" << expiry;
            }
        }
    }
}

// Volga is negative near the money (where d1 and d2 straddle zero) and positive
// in both wings. That sign flip is the analytical statement of why a butterfly
// is long vol-of-vol, so it is worth pinning down rather than only checking a
// finite difference.
TEST(Greeks, VolgaChangesSignThroughTheMoney) {
    const opt::MarketData market{100.0, 0.0, 0.0, 0.20};
    EXPECT_LT(opt::bs_volga(market, 100.0, 1.0), 0.0);
    EXPECT_GT(opt::bs_volga(market, 60.0, 1.0), 0.0);
    EXPECT_GT(opt::bs_volga(market, 170.0, 1.0), 0.0);
}

// Vanna carries the sign of -d2, so it is positive for a strike above the
// forward and negative below it. Read through delta: more volatility pulls
// every call's delta towards 0.5, which means *up* for an out-of-the-money call
// and *down* for an in-the-money one.
TEST(Greeks, VannaChangesSignThroughTheForward) {
    const opt::MarketData market{100.0, 0.0, 0.0, 0.20};
    EXPECT_GT(opt::bs_vanna(market, 130.0, 1.0), 0.0);
    EXPECT_LT(opt::bs_vanna(market, 75.0, 1.0), 0.0);

    // The crossing is where d2 = 0, i.e. K = F exp(-sigma^2 T / 2), which sits
    // slightly *below* the forward rather than on it.
    const double zero_vanna_strike = 100.0 * std::exp(-0.5 * 0.20 * 0.20 * 1.0);
    EXPECT_NEAR(opt::bs_vanna(market, zero_vanna_strike, 1.0), 0.0, 1e-12);
}

// Both vanish where there is no uncertainty left to be second-order about.
TEST(Greeks, SecondOrderGreeksVanishInDegenerateLimits) {
    const opt::MarketData zero_vol{100.0, 0.04, 0.0, 0.0};
    EXPECT_DOUBLE_EQ(opt::bs_vanna(zero_vol, 100.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(opt::bs_volga(zero_vol, 100.0, 1.0), 0.0);

    const opt::MarketData market{100.0, 0.04, 0.0, 0.20};
    EXPECT_DOUBLE_EQ(opt::bs_vanna(market, 100.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(opt::bs_volga(market, 100.0, 0.0), 0.0);
}
