// Static no-arbitrage properties of the pricing engine.
//
// Each test here encodes a relationship that must hold for *any* arbitrage-free
// model, not just for Black-Scholes: they are model-independent statements
// about what an option contract is worth. A pricer that violates one of them is
// quotable against, which is a different and more serious kind of wrong than
// being a few basis points off a reference value.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "opt/binomial_tree.hpp"
#include "opt/black_scholes.hpp"
#include "opt/greeks.hpp"
#include "opt/market_data.hpp"
#include "opt/option.hpp"

namespace {

using opt::MarketData;
using opt::OptionType;

// A spread of markets covering low/high vol, positive and negative rates, with
// and without a dividend yield, and short and long maturities.
struct Scenario {
    MarketData market;
    double strike;
    double expiry;
};

[[nodiscard]] std::vector<Scenario> scenarios() {
    std::vector<Scenario> out;
    for (const double spot : {50.0, 100.0, 150.0}) {
        for (const double vol : {0.05, 0.20, 0.80}) {
            for (const double rate : {-0.01, 0.0, 0.05}) {
                for (const double div : {0.0, 0.03}) {
                    for (const double strike : {60.0, 100.0, 140.0}) {
                        for (const double expiry : {0.02, 0.5, 3.0}) {
                            out.push_back({MarketData{spot, rate, div, vol}, strike, expiry});
                        }
                    }
                }
            }
        }
    }
    return out;
}

}  // namespace

// -----------------------------------------------------------------------------
// Put-call parity
// -----------------------------------------------------------------------------

// C - P = S e^{-qT} - K e^{-rT} is a portfolio identity, not a model result: it
// follows from the two sides having identical payoffs at expiry. It should hold
// to rounding error, so the tolerance is scaled to the magnitude of the terms
// being differenced rather than fixed.
TEST(NoArbitrage, PutCallParityHoldsToMachinePrecision) {
    for (const auto& s : scenarios()) {
        const double call = opt::black_scholes_call(s.market, s.strike, s.expiry);
        const double put = opt::black_scholes_put(s.market, s.strike, s.expiry);
        const double forward = s.market.spot * std::exp(-s.market.dividend * s.expiry);
        const double pv_strike = s.strike * std::exp(-s.market.rate * s.expiry);

        const double scale = std::max({call, put, forward, pv_strike, 1.0});
        EXPECT_NEAR(call - put, forward - pv_strike, 1e-13 * scale)
            << "S=" << s.market.spot << " K=" << s.strike << " T=" << s.expiry
            << " vol=" << s.market.volatility;
    }
}

// -----------------------------------------------------------------------------
// Static price bounds
// -----------------------------------------------------------------------------

// A European call is worth at least its forward intrinsic value (otherwise buy
// the call and sell the forward) and never more than the dividend-discounted
// spot (otherwise buy the stock and sell the call).
TEST(NoArbitrage, EuropeanCallRespectsStaticBounds) {
    for (const auto& s : scenarios()) {
        const double call = opt::black_scholes_call(s.market, s.strike, s.expiry);
        const double forward = s.market.spot * std::exp(-s.market.dividend * s.expiry);
        const double pv_strike = s.strike * std::exp(-s.market.rate * s.expiry);
        const double lower = std::max(forward - pv_strike, 0.0);

        EXPECT_GE(call, lower - 1e-12 * std::max(forward, 1.0));
        EXPECT_LE(call, forward + 1e-12 * std::max(forward, 1.0));
    }
}

// The mirror image for puts: at least the discounted intrinsic, at most the
// present value of the strike (the most a put can ever pay).
TEST(NoArbitrage, EuropeanPutRespectsStaticBounds) {
    for (const auto& s : scenarios()) {
        const double put = opt::black_scholes_put(s.market, s.strike, s.expiry);
        const double forward = s.market.spot * std::exp(-s.market.dividend * s.expiry);
        const double pv_strike = s.strike * std::exp(-s.market.rate * s.expiry);
        const double lower = std::max(pv_strike - forward, 0.0);

        EXPECT_GE(put, lower - 1e-12 * std::max(pv_strike, 1.0));
        EXPECT_LE(put, pv_strike + 1e-12 * std::max(pv_strike, 1.0));
    }
}

// -----------------------------------------------------------------------------
// Monotonicity
// -----------------------------------------------------------------------------

TEST(NoArbitrage, CallIncreasesInSpotAndPutDecreases) {
    for (const auto& s : scenarios()) {
        MarketData up = s.market;
        up.spot = s.market.spot * 1.01;
        EXPECT_GT(opt::black_scholes_call(up, s.strike, s.expiry),
                  opt::black_scholes_call(s.market, s.strike, s.expiry) - 1e-15);
        EXPECT_LT(opt::black_scholes_put(up, s.strike, s.expiry),
                  opt::black_scholes_put(s.market, s.strike, s.expiry) + 1e-15);
    }
}

TEST(NoArbitrage, CallDecreasesInStrikeAndPutIncreases) {
    for (const auto& s : scenarios()) {
        const double higher = s.strike * 1.01;
        EXPECT_LT(opt::black_scholes_call(s.market, higher, s.expiry),
                  opt::black_scholes_call(s.market, s.strike, s.expiry) + 1e-15);
        EXPECT_GT(opt::black_scholes_put(s.market, higher, s.expiry),
                  opt::black_scholes_put(s.market, s.strike, s.expiry) - 1e-15);
    }
}

// Vega is positive for both calls and puts, so neither can fall when vol rises.
//
// The assertion is non-strict, and that is not a hedge. For a deep in-the-money
// short-dated option the price is essentially the deterministic forward value
// and vega underflows: bumping vol by a full point moves a price of ~90 by less
// than one unit in the last place, so the two doubles compare equal. The strict
// statement is only available where vega is numerically resolvable, so it is
// asserted separately and only there.
TEST(NoArbitrage, NeitherTypeFallsWhenVolatilityRises) {
    for (const auto& s : scenarios()) {
        MarketData up = s.market;
        up.volatility = s.market.volatility + 0.01;

        for (const OptionType type : {OptionType::Call, OptionType::Put}) {
            const double base = opt::black_scholes_price(s.market, s.strike, s.expiry, type);
            const double bumped = opt::black_scholes_price(up, s.strike, s.expiry, type);
            const double slack = 8.0 * std::numeric_limits<double>::epsilon() * std::max(base, 1.0);
            EXPECT_GE(bumped, base - slack) << "S=" << s.market.spot << " K=" << s.strike
                                            << " T=" << s.expiry << " vol=" << s.market.volatility;

            // 0.01 of vega is the first-order price change from the bump; ask
            // for it to clear a hundred ulps before demanding a strict increase.
            const double expected_move = 0.01 * opt::bs_vega(s.market, s.strike, s.expiry);
            if (expected_move > 100.0 * slack) {
                EXPECT_GT(bumped, base);
            }
        }
    }
}

// Monotonicity in maturity needs a caveat. A longer-dated European call on a
// non-dividend-paying stock is worth more (the extra time is pure optionality
// plus a later strike payment), but with a dividend yield a long-dated call can
// be worth *less* because the holder forgoes more carry. The test therefore
// fixes q = 0 and r >= 0, which is exactly the regime where the result holds.
TEST(NoArbitrage, CallDoesNotFallWithMaturityWithoutDividends) {
    for (const auto& s : scenarios()) {
        if (s.market.dividend != 0.0 || s.market.rate < 0.0) {
            continue;
        }
        const double base = opt::black_scholes_call(s.market, s.strike, s.expiry);
        const double longer = opt::black_scholes_call(s.market, s.strike, s.expiry * 1.05);
        // Same underflow caveat as the volatility test: deep in the money and
        // short dated, the extra 5% of maturity is worth less than one ulp.
        const double slack = 8.0 * std::numeric_limits<double>::epsilon() * std::max(base, 1.0);
        EXPECT_GE(longer, base - slack)
            << "S=" << s.market.spot << " K=" << s.strike << " T=" << s.expiry;
    }
}

// -----------------------------------------------------------------------------
// Convexity in strike (butterfly)
// -----------------------------------------------------------------------------

// A butterfly -- long one K-dK call, short two K calls, long one K+dK call --
// has a non-negative payoff in every state, so it cannot cost less than zero.
// Equivalently, the price is convex in strike.
TEST(NoArbitrage, PriceIsConvexInStrike) {
    for (const auto& s : scenarios()) {
        const double wing = 5.0;
        if (s.strike - wing <= 0.0) {
            continue;
        }
        for (const OptionType type : {OptionType::Call, OptionType::Put}) {
            const double low = opt::black_scholes_price(s.market, s.strike - wing, s.expiry, type);
            const double mid = opt::black_scholes_price(s.market, s.strike, s.expiry, type);
            const double high = opt::black_scholes_price(s.market, s.strike + wing, s.expiry, type);
            EXPECT_GE(low - 2.0 * mid + high, -1e-12 * std::max(mid, 1.0))
                << "butterfly at K=" << s.strike << " T=" << s.expiry;
        }
    }
}

// -----------------------------------------------------------------------------
// American versus European
// -----------------------------------------------------------------------------

// Early exercise is an extra right, never an obligation, so the American price
// can never be below the European one on the same lattice.
TEST(NoArbitrage, AmericanPutIsNeverCheaperThanEuropean) {
    constexpr int kSteps = 400;
    for (const auto& s : scenarios()) {
        const opt::EuropeanPut euro{s.strike, s.expiry};
        const opt::AmericanPut amer{s.strike, s.expiry};
        EXPECT_GE(opt::binomial_price(amer, s.market, kSteps),
                  opt::binomial_price(euro, s.market, kSteps) - 1e-12);
    }
}

// Merton's result: on a non-dividend-paying underlying with a non-negative rate
// it is never optimal to exercise an American call early, so the two prices
// coincide.
//
// How exactly they coincide depends on the rate, and the distinction is worth
// spelling out because it is easy to write a test that passes for the wrong
// reason. With r > 0 the continuation value strictly dominates the intrinsic at
// every node, the max() always takes the same branch, and the two prices are
// bit-for-bit identical. With r == 0 exactly the two are mathematically tied at
// deep in-the-money nodes, so which side the max() takes is decided by rounding
// -- and the early-exercise branch is in fact the more accurate of the two
// (it returns 40 where the rolled-back value has drifted to 39.999999999998977).
// The prices then agree only to within accumulated rounding, which is what the
// relative tolerance below allows.
TEST(NoArbitrage, AmericanCallEqualsEuropeanWithoutDividends) {
    constexpr int kSteps = 400;
    for (const auto& s : scenarios()) {
        if (s.market.dividend != 0.0 || s.market.rate < 0.0) {
            continue;
        }
        const opt::EuropeanCall euro{s.strike, s.expiry};
        const opt::AmericanCall amer{s.strike, s.expiry};
        const double european = opt::binomial_price(euro, s.market, kSteps);
        const double american = opt::binomial_price(amer, s.market, kSteps);

        // Node-wise max() is monotone, so this direction is exact regardless.
        EXPECT_GE(american, european);

        if (s.market.rate > 0.0) {
            EXPECT_DOUBLE_EQ(american, european);
        } else {
            EXPECT_NEAR(american, european, 1e-12 * std::max(european, 1.0))
                << "S=" << s.market.spot << " K=" << s.strike << " T=" << s.expiry;
        }
    }
}

// With a dividend yield large enough to make the carry negative, early exercise
// of a deep in-the-money American call becomes optimal and the premium is
// strictly positive. This is the converse of the test above: it shows the
// lattice is genuinely testing the exercise boundary rather than never taking
// the early-exercise branch at all.
TEST(NoArbitrage, AmericanCallPremiumIsPositiveUnderLargeDividends) {
    const MarketData market{100.0, 0.02, 0.15, 0.25};
    const opt::EuropeanCall euro{60.0, 2.0};
    const opt::AmericanCall amer{60.0, 2.0};
    EXPECT_GT(opt::binomial_price(amer, market, 800) - opt::binomial_price(euro, market, 800),
              1e-3);
}

// -----------------------------------------------------------------------------
// Limiting cases
// -----------------------------------------------------------------------------

// With no diffusion the payoff is certain: the option is a forward struck at K,
// worth max(S e^{-qT} - K e^{-rT}, 0). Note this is *not* the discounted spot
// intrinsic max(S - K, 0) e^{-rT} -- carry matters.
TEST(NoArbitrage, ZeroVolatilityGivesDiscountedForwardIntrinsic) {
    for (const auto& s : scenarios()) {
        MarketData market = s.market;
        market.volatility = 0.0;
        const double forward = market.spot * std::exp(-market.dividend * s.expiry);
        const double pv_strike = s.strike * std::exp(-market.rate * s.expiry);
        EXPECT_NEAR(opt::black_scholes_call(market, s.strike, s.expiry),
                    std::max(forward - pv_strike, 0.0), 1e-12);
        EXPECT_NEAR(opt::black_scholes_put(market, s.strike, s.expiry),
                    std::max(pv_strike - forward, 0.0), 1e-12);
    }
}

// At expiry the option is worth its intrinsic value exactly.
TEST(NoArbitrage, ZeroTimeGivesIntrinsic) {
    for (const auto& s : scenarios()) {
        EXPECT_NEAR(opt::black_scholes_call(s.market, s.strike, 0.0),
                    std::max(s.market.spot - s.strike, 0.0), 1e-12);
        EXPECT_NEAR(opt::black_scholes_put(s.market, s.strike, 0.0),
                    std::max(s.strike - s.market.spot, 0.0), 1e-12);
    }
}

// The limit is approached continuously, not just attained at T = 0 exactly.
TEST(NoArbitrage, PriceApproachesIntrinsicAsExpiryShrinks) {
    const MarketData market{100.0, 0.05, 0.01, 0.30};
    double previous = 1e9;
    for (const double expiry : {1e-2, 1e-3, 1e-4, 1e-5, 1e-6}) {
        const double gap = opt::black_scholes_call(market, 90.0, expiry) - 10.0;
        EXPECT_LT(std::abs(gap), std::abs(previous));
        previous = gap;
    }
    EXPECT_LT(std::abs(previous), 1e-4);
}

// Deep out of the money in units of total volatility sigma*sqrt(T), the price
// collapses to zero; deep in the money it collapses to the forward value, and
// delta saturates at the two ends of its range.
TEST(NoArbitrage, DeepMoneynessAsymptotics) {
    const MarketData market{100.0, 0.04, 0.02, 0.20};
    const double expiry = 1.0;
    const double forward = market.spot * std::exp(-market.dividend * expiry);

    const double deep_otm_strike = 100000.0;
    EXPECT_NEAR(opt::black_scholes_call(market, deep_otm_strike, expiry), 0.0, 1e-12);
    EXPECT_NEAR(opt::bs_delta(market, deep_otm_strike, expiry, OptionType::Call), 0.0, 1e-12);

    const double deep_itm_strike = 0.01;
    const double pv_strike = deep_itm_strike * std::exp(-market.rate * expiry);
    EXPECT_NEAR(opt::black_scholes_call(market, deep_itm_strike, expiry), forward - pv_strike,
                1e-10);
    EXPECT_NEAR(opt::bs_delta(market, deep_itm_strike, expiry, OptionType::Call),
                std::exp(-market.dividend * expiry), 1e-12);

    EXPECT_NEAR(opt::black_scholes_put(market, deep_itm_strike, expiry), 0.0, 1e-12);
    EXPECT_NEAR(opt::bs_delta(market, deep_otm_strike, expiry, OptionType::Put),
                -std::exp(-market.dividend * expiry), 1e-12);
}

// Gamma and vega vanish in both deep tails: there is nothing left to be
// uncertain about, so the price stops responding to a change in volatility.
TEST(NoArbitrage, GammaAndVegaVanishInBothTails) {
    const MarketData market{100.0, 0.04, 0.02, 0.20};
    for (const double strike : {0.01, 100000.0}) {
        EXPECT_NEAR(opt::bs_gamma(market, strike, 1.0), 0.0, 1e-12);
        EXPECT_NEAR(opt::bs_vega(market, strike, 1.0), 0.0, 1e-12);
    }
}
