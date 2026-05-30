// Unit tests for the Phase 1 core data types: MarketData and the Option
// hierarchy (construction, accessors, payoffs, exercise/type tags, cloning,
// and input validation).
#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

#include "opt/market_data.hpp"
#include "opt/option.hpp"

namespace {

TEST(MarketData, AggregateInitialisationAndValidity) {
    const opt::MarketData mkt{/*spot=*/100.0, /*rate=*/0.05, /*dividend=*/0.01,
                              /*volatility=*/0.2};
    EXPECT_DOUBLE_EQ(mkt.spot, 100.0);
    EXPECT_DOUBLE_EQ(mkt.rate, 0.05);
    EXPECT_DOUBLE_EQ(mkt.dividend, 0.01);
    EXPECT_DOUBLE_EQ(mkt.volatility, 0.2);
    EXPECT_TRUE(mkt.is_valid());
}

TEST(MarketData, NegativeSpotOrVolIsInvalidButNegativeRateIsFine) {
    EXPECT_FALSE((opt::MarketData{-1.0, 0.05, 0.0, 0.2}).is_valid());
    EXPECT_FALSE((opt::MarketData{100.0, 0.05, 0.0, -0.2}).is_valid());
    EXPECT_TRUE((opt::MarketData{100.0, -0.01, 0.0, 0.2}).is_valid());
}

TEST(Option, AccessorsAndTags) {
    const opt::EuropeanCall call{100.0, 1.0};
    EXPECT_DOUBLE_EQ(call.strike(), 100.0);
    EXPECT_DOUBLE_EQ(call.expiry(), 1.0);
    EXPECT_EQ(call.type(), opt::OptionType::Call);
    EXPECT_EQ(call.exercise(), opt::Exercise::European);
    EXPECT_TRUE(call.is_call());
    EXPECT_FALSE(call.is_american());

    const opt::AmericanPut put{100.0, 0.5};
    EXPECT_EQ(put.type(), opt::OptionType::Put);
    EXPECT_EQ(put.exercise(), opt::Exercise::American);
    EXPECT_FALSE(put.is_call());
    EXPECT_TRUE(put.is_american());
}

TEST(Option, CallPayoff) {
    const opt::EuropeanCall call{100.0, 1.0};
    EXPECT_DOUBLE_EQ(call.payoff(120.0), 20.0);  // in the money
    EXPECT_DOUBLE_EQ(call.payoff(100.0), 0.0);   // at the money
    EXPECT_DOUBLE_EQ(call.payoff(80.0), 0.0);    // out of the money, clipped
}

TEST(Option, PutPayoff) {
    const opt::EuropeanPut put{100.0, 1.0};
    EXPECT_DOUBLE_EQ(put.payoff(80.0), 20.0);
    EXPECT_DOUBLE_EQ(put.payoff(100.0), 0.0);
    EXPECT_DOUBLE_EQ(put.payoff(120.0), 0.0);
}

TEST(Option, PayoffIsAccessibleThroughBasePointer) {
    const std::unique_ptr<opt::Option> opt = std::make_unique<opt::AmericanCall>(50.0, 2.0);
    EXPECT_EQ(opt->type(), opt::OptionType::Call);
    EXPECT_DOUBLE_EQ(opt->payoff(60.0), 10.0);
}

TEST(Option, CloneProducesIndependentEquivalentCopy) {
    const opt::EuropeanPut original{100.0, 1.0};
    const std::unique_ptr<opt::Option> copy = original.clone();

    ASSERT_NE(copy.get(), nullptr);
    EXPECT_NE(copy.get(), static_cast<const opt::Option*>(&original));
    EXPECT_DOUBLE_EQ(copy->strike(), original.strike());
    EXPECT_DOUBLE_EQ(copy->expiry(), original.expiry());
    EXPECT_EQ(copy->type(), original.type());
    EXPECT_EQ(copy->exercise(), original.exercise());
    // The dynamic type is preserved, not sliced to the base.
    EXPECT_NE(dynamic_cast<opt::EuropeanPut*>(copy.get()), nullptr);
}

TEST(Option, InvalidTermsThrow) {
    EXPECT_THROW(opt::EuropeanCall(0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(opt::EuropeanCall(-10.0, 1.0), std::invalid_argument);
    EXPECT_THROW(opt::EuropeanCall(100.0, -1.0), std::invalid_argument);
    EXPECT_NO_THROW(opt::EuropeanCall(100.0, 0.0));  // expiry == 0 is allowed
}

}  // namespace
