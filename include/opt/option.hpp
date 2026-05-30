#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace opt {

/// Whether an option is a call (right to buy) or a put (right to sell).
enum class OptionType { Call, Put };

/// When an option may be exercised: only at expiry (European) or at any time up
/// to expiry (American).
enum class Exercise { European, American };

/// Abstract base for an option contract. An option owns its economic terms —
/// the strike and the time to expiry (in years) — while market observables
/// (spot, rate, dividend, volatility) are supplied separately via
/// opt::MarketData at pricing time.
///
/// The hierarchy is polymorphic, so copy/move are made protected to prevent
/// object slicing; use clone() to duplicate through a base pointer
/// (C++ Core Guidelines C.67).
class Option {
public:
    virtual ~Option() = default;

    /// Strike price \f$K > 0\f$.
    [[nodiscard]] double strike() const noexcept { return strike_; }

    /// Time to expiry in years, \f$T \ge 0\f$.
    [[nodiscard]] double expiry() const noexcept { return expiry_; }

    /// Call or put.
    [[nodiscard]] virtual OptionType type() const noexcept = 0;

    /// European or American exercise.
    [[nodiscard]] virtual Exercise exercise() const noexcept = 0;

    /// True for a call, false for a put.
    [[nodiscard]] bool is_call() const noexcept { return type() == OptionType::Call; }

    /// True for American exercise, false for European.
    [[nodiscard]] bool is_american() const noexcept {
        return exercise() == Exercise::American;
    }

    /// Intrinsic value (the immediate-exercise payoff) for a given underlying
    /// price: \f$\max(S-K,0)\f$ for a call, \f$\max(K-S,0)\f$ for a put.
    [[nodiscard]] double payoff(double underlying) const noexcept {
        return is_call() ? std::max(underlying - strike_, 0.0)
                         : std::max(strike_ - underlying, 0.0);
    }

    /// Polymorphic deep copy.
    [[nodiscard]] virtual std::unique_ptr<Option> clone() const = 0;

protected:
    /// @throws std::invalid_argument if @p strike is not positive or @p expiry
    ///         is negative.
    Option(double strike, double expiry) : strike_{strike}, expiry_{expiry} {
        if (!(strike > 0.0)) {
            throw std::invalid_argument("Option: strike must be positive");
        }
        if (!(expiry >= 0.0)) {
            throw std::invalid_argument("Option: expiry must be non-negative");
        }
    }

    Option(const Option&) = default;
    Option(Option&&) = default;
    Option& operator=(const Option&) = default;
    Option& operator=(Option&&) = default;

private:
    double strike_;
    double expiry_;
};

/// European call: exercisable only at expiry; payoff \f$\max(S_T - K, 0)\f$.
class EuropeanCall final : public Option {
public:
    EuropeanCall(double strike, double expiry) : Option(strike, expiry) {}

    [[nodiscard]] OptionType type() const noexcept override { return OptionType::Call; }
    [[nodiscard]] Exercise exercise() const noexcept override { return Exercise::European; }
    [[nodiscard]] std::unique_ptr<Option> clone() const override {
        return std::make_unique<EuropeanCall>(*this);
    }
};

/// European put: exercisable only at expiry; payoff \f$\max(K - S_T, 0)\f$.
class EuropeanPut final : public Option {
public:
    EuropeanPut(double strike, double expiry) : Option(strike, expiry) {}

    [[nodiscard]] OptionType type() const noexcept override { return OptionType::Put; }
    [[nodiscard]] Exercise exercise() const noexcept override { return Exercise::European; }
    [[nodiscard]] std::unique_ptr<Option> clone() const override {
        return std::make_unique<EuropeanPut>(*this);
    }
};

/// American call: exercisable at any time up to expiry; call payoff.
class AmericanCall final : public Option {
public:
    AmericanCall(double strike, double expiry) : Option(strike, expiry) {}

    [[nodiscard]] OptionType type() const noexcept override { return OptionType::Call; }
    [[nodiscard]] Exercise exercise() const noexcept override { return Exercise::American; }
    [[nodiscard]] std::unique_ptr<Option> clone() const override {
        return std::make_unique<AmericanCall>(*this);
    }
};

/// American put: exercisable at any time up to expiry; put payoff.
class AmericanPut final : public Option {
public:
    AmericanPut(double strike, double expiry) : Option(strike, expiry) {}

    [[nodiscard]] OptionType type() const noexcept override { return OptionType::Put; }
    [[nodiscard]] Exercise exercise() const noexcept override { return Exercise::American; }
    [[nodiscard]] std::unique_ptr<Option> clone() const override {
        return std::make_unique<AmericanPut>(*this);
    }
};

}  // namespace opt
