// price_cli -- command-line front-end to the pricing engine.
//
// Examples, pricing then inverting the same contract:
//   price_cli --spot 100 --strike 105 --rate 0.05 --vol 0.2 --expiry 1 --method bs
//   price_cli --spot 100 --strike 105 --rate 0.05 --implied-vol 8.02 --expiry 1 --method bs
//
// The tool runs in one of two directions. Given --vol it prices the option;
// given --implied-vol it takes a market price and solves for the volatility
// that reproduces it. Either way it prints the analytical Black-Scholes
// Greeks -- in the inversion case, at the volatility that was recovered --
// plus, for Monte Carlo, the standard error and 95% confidence interval.
// The argument parser is hand-rolled; there are no external dependencies
// beyond the engine itself.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "opt/binomial_tree.hpp"
#include "opt/black_scholes.hpp"
#include "opt/greeks.hpp"
#include "opt/implied_volatility.hpp"
#include "opt/market_data.hpp"
#include "opt/monte_carlo.hpp"
#include "opt/option.hpp"

namespace {

enum class Method { BlackScholes, Tree, MonteCarlo };

struct Config {
    double spot = 0.0;
    double strike = 0.0;
    double rate = 0.0;
    double dividend = 0.0;
    double volatility = 0.0;
    double expiry = 0.0;
    opt::OptionType type = opt::OptionType::Call;
    opt::Exercise exercise = opt::Exercise::European;
    Method method = Method::BlackScholes;
    int steps = 1000;                     // tree
    std::size_t paths = 100'000;          // Monte Carlo
    std::uint64_t seed = 0x5DEECE66DULL;  // Monte Carlo
    opt::VarianceReduction variates = opt::VarianceReduction::AntitheticAndControl;

    // Inversion mode: solve for the volatility reproducing target_price
    // instead of pricing at a volatility supplied on the command line.
    bool solve_iv = false;
    double target_price = 0.0;
};

void print_usage(std::ostream& os) {
    os << "Usage: price_cli [options]\n\n"
          "Required:\n"
          "  --spot <S>        underlying spot price (> 0)\n"
          "  --strike <K>      strike price (> 0)\n"
          "  --rate <r>        continuously-compounded risk-free rate\n"
          "  --expiry <T>      time to expiry in years (>= 0)\n"
          "  and exactly one of:\n"
          "  --vol <sigma>     annualised volatility (>= 0): prices the option\n"
          "  --implied-vol <P> market price (>= 0): solves for the volatility\n"
          "                    reproducing it (--method bs or tree; needs T > 0)\n\n"
          "Optional:\n"
          "  --dividend <q>    continuous dividend yield (default 0)\n"
          "  --type <call|put>            (default call)\n"
          "  --exercise <european|american>  (default european; tree only)\n"
          "  --method <bs|tree|mc>        (default bs)\n"
          "  --steps <N>       tree steps (default 1000)\n"
          "  --paths <N>       Monte Carlo paths (default 100000)\n"
          "  --seed <N>        Monte Carlo RNG seed\n"
          "  --variates <none|antithetic|control|both>  (default both)\n"
          "  -h, --help        show this help\n";
}

[[nodiscard]] double to_double(std::string_view flag, const std::string& value) {
    try {
        std::size_t consumed = 0;
        const double parsed = std::stod(value, &consumed);
        if (consumed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::runtime_error("invalid number for " + std::string(flag) + ": '" + value + "'");
    }
}

[[nodiscard]] long long to_int(std::string_view flag, const std::string& value) {
    try {
        std::size_t consumed = 0;
        const long long parsed = std::stoll(value, &consumed);
        if (consumed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::runtime_error("invalid integer for " + std::string(flag) + ": '" + value + "'");
    }
}

[[nodiscard]] opt::OptionType parse_type(const std::string& v) {
    if (v == "call") {
        return opt::OptionType::Call;
    }
    if (v == "put") {
        return opt::OptionType::Put;
    }
    throw std::runtime_error("--type must be 'call' or 'put', got '" + v + "'");
}

[[nodiscard]] opt::Exercise parse_exercise(const std::string& v) {
    if (v == "european") {
        return opt::Exercise::European;
    }
    if (v == "american") {
        return opt::Exercise::American;
    }
    throw std::runtime_error("--exercise must be 'european' or 'american', got '" + v + "'");
}

[[nodiscard]] Method parse_method(const std::string& v) {
    if (v == "bs") {
        return Method::BlackScholes;
    }
    if (v == "tree") {
        return Method::Tree;
    }
    if (v == "mc") {
        return Method::MonteCarlo;
    }
    throw std::runtime_error("--method must be 'bs', 'tree' or 'mc', got '" + v + "'");
}

[[nodiscard]] opt::VarianceReduction parse_variates(const std::string& v) {
    if (v == "none") {
        return opt::VarianceReduction::None;
    }
    if (v == "antithetic") {
        return opt::VarianceReduction::Antithetic;
    }
    if (v == "control") {
        return opt::VarianceReduction::ControlVariate;
    }
    if (v == "both") {
        return opt::VarianceReduction::AntitheticAndControl;
    }
    throw std::runtime_error("--variates must be none|antithetic|control|both, got '" + v + "'");
}

// Rules that need the whole command line rather than one flag, so they run once
// parsing is done. Throws std::runtime_error describing the first violation.
void validate(const Config& cfg, bool have_required, bool have_vol) {
    if (!have_required) {
        throw std::runtime_error("missing required flag(s); need --spot --strike --rate --expiry");
    }
    // The volatility is either an input or the unknown, never both.
    if (have_vol && cfg.solve_iv) {
        throw std::runtime_error(
            "--vol and --implied-vol are mutually exclusive: --implied-vol solves "
            "for the volatility, so there is nothing to supply");
    }
    if (!have_vol && !cfg.solve_iv) {
        throw std::runtime_error("need either --vol (to price) or --implied-vol (to invert)");
    }
    if (cfg.solve_iv) {
        if (cfg.method == Method::MonteCarlo) {
            throw std::runtime_error(
                "--implied-vol needs --method bs or tree: a Monte Carlo price carries "
                "sampling noise, so inverting it would solve for the noise as well");
        }
        if (cfg.target_price < 0.0) {
            throw std::runtime_error("--implied-vol must be >= 0");
        }
        if (cfg.expiry <= 0.0) {
            throw std::runtime_error(
                "--implied-vol needs --expiry > 0: at expiry the price is the payoff and "
                "carries no volatility information");
        }
    }
    if (cfg.spot <= 0.0) {
        throw std::runtime_error("--spot must be > 0");
    }
    if (cfg.strike <= 0.0) {
        throw std::runtime_error("--strike must be > 0");
    }
    if (cfg.volatility < 0.0) {
        throw std::runtime_error("--vol must be >= 0");
    }
    if (cfg.expiry < 0.0) {
        throw std::runtime_error("--expiry must be >= 0");
    }
    if (cfg.method == Method::Tree && cfg.steps < 1) {
        throw std::runtime_error("--steps must be >= 1");
    }
}

// Parses argv into a Config. Returns false (after printing usage) when --help
// was requested; throws std::runtime_error on a malformed command line.
[[nodiscard]] bool parse_args(int argc, char** argv, Config& cfg) {
    bool have_spot = false;
    bool have_strike = false;
    bool have_rate = false;
    bool have_vol = false;
    bool have_expiry = false;

    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "-h" || flag == "--help") {
            print_usage(std::cout);
            return false;
        }
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + flag);
            }
            return argv[++i];
        };

        if (flag == "--spot") {
            cfg.spot = to_double(flag, value());
            have_spot = true;
        } else if (flag == "--strike") {
            cfg.strike = to_double(flag, value());
            have_strike = true;
        } else if (flag == "--rate") {
            cfg.rate = to_double(flag, value());
            have_rate = true;
        } else if (flag == "--vol") {
            cfg.volatility = to_double(flag, value());
            have_vol = true;
        } else if (flag == "--implied-vol") {
            cfg.target_price = to_double(flag, value());
            cfg.solve_iv = true;
        } else if (flag == "--expiry") {
            cfg.expiry = to_double(flag, value());
            have_expiry = true;
        } else if (flag == "--dividend") {
            cfg.dividend = to_double(flag, value());
        } else if (flag == "--type") {
            cfg.type = parse_type(value());
        } else if (flag == "--exercise") {
            cfg.exercise = parse_exercise(value());
        } else if (flag == "--method") {
            cfg.method = parse_method(value());
        } else if (flag == "--steps") {
            cfg.steps = static_cast<int>(to_int(flag, value()));
        } else if (flag == "--paths") {
            cfg.paths = static_cast<std::size_t>(to_int(flag, value()));
        } else if (flag == "--seed") {
            cfg.seed = static_cast<std::uint64_t>(to_int(flag, value()));
        } else if (flag == "--variates") {
            cfg.variates = parse_variates(value());
        } else {
            throw std::runtime_error("unknown flag: " + flag);
        }
    }

    validate(cfg, have_spot && have_strike && have_rate && have_expiry, have_vol);
    return true;
}

[[nodiscard]] std::unique_ptr<opt::Option> make_option(const Config& cfg) {
    const bool call = cfg.type == opt::OptionType::Call;
    if (cfg.exercise == opt::Exercise::American) {
        if (call) {
            return std::make_unique<opt::AmericanCall>(cfg.strike, cfg.expiry);
        }
        return std::make_unique<opt::AmericanPut>(cfg.strike, cfg.expiry);
    }
    if (call) {
        return std::make_unique<opt::EuropeanCall>(cfg.strike, cfg.expiry);
    }
    return std::make_unique<opt::EuropeanPut>(cfg.strike, cfg.expiry);
}

const char* method_name(Method m) {
    switch (m) {
        case Method::BlackScholes:
            return "Black-Scholes (analytical)";
        case Method::Tree:
            return "CRR binomial tree";
        case Method::MonteCarlo:
            return "Monte Carlo";
    }
    return "unknown";
}

const char* variates_name(opt::VarianceReduction v) {
    switch (v) {
        case opt::VarianceReduction::None:
            return "none";
        case opt::VarianceReduction::Antithetic:
            return "antithetic";
        case opt::VarianceReduction::ControlVariate:
            return "control";
        case opt::VarianceReduction::AntitheticAndControl:
            return "antithetic+control";
    }
    return "unknown";
}

// Inverts the quote under the model the caller chose. The lattice path exists
// because a listed equity option is American: inverting its quote with
// Black-Scholes prices away the early-exercise premium and calls the difference
// volatility.
[[nodiscard]] opt::IvResult solve_implied_volatility(const Config& cfg,
                                                     const opt::MarketData& market) {
    if (cfg.method == Method::Tree) {
        const std::unique_ptr<opt::Option> option = make_option(cfg);
        return opt::implied_volatility_binomial(cfg.target_price, market, *option, cfg.steps);
    }
    return opt::implied_volatility(cfg.target_price, market, cfg.strike, cfg.expiry, cfg.type);
}

// A failed inversion is nearly always a statement about the quote rather than
// about the solver, so say which.
const char* iv_diagnosis(opt::IvStatus status) {
    switch (status) {
        case opt::IvStatus::Converged:
            return "";
        case opt::IvStatus::BelowIntrinsic:
            return "the quote is below the option's intrinsic value, so no volatility "
                   "reproduces it -- check the spot, rate and dividend used";
        case opt::IvStatus::AboveUpperBound:
            return "the quote exceeds what any volatility can produce (the price is bounded "
                   "by the discounted forward), so it is not an arbitrage-free quote";
        case opt::IvStatus::MaxIterations:
            return "the solver ran out of iterations, which on a well-posed quote means the "
                   "tolerances are below what double precision can deliver here";
        case opt::IvStatus::InvalidInput:
            return "the inputs are not a well-posed inversion problem";
        case opt::IvStatus::BelowLatticeResolution:
            return "the quote sits between the zero-volatility price and the cheapest price "
                   "this lattice can produce, so it implies no volatility this tree can "
                   "represent -- it is indistinguishable from a deterministic forward here";
    }
    return "unknown failure";
}

// Prints the recovered volatility together with the conditioning of the
// inversion. The vega matters as much as the answer: it converts a quote error
// into a volatility error, and where it collapses the number is arithmetic
// without economics.
void print_implied_volatility(const opt::IvResult& iv) {
    std::cout << "Implied vol   : " << iv.volatility << '\n'
              << "  Vega        : " << iv.vega << "   (per 1.00 vol)\n"
              << "  Residual    : " << iv.price_residual << "   (model - market)\n"
              << "  Solver      : " << opt::to_string(iv.method) << ", " << iv.iterations
              << " iterations\n";

    constexpr double negligible_vega = 1e-12;
    if (!(std::abs(iv.vega) > negligible_vega)) {
        std::cout << "  WARNING     : the price is flat in volatility here (vega ~ 0), so the\n"
                     "                implied volatility is not identified -- every volatility\n"
                     "                in a whole interval reproduces this quote. Discard it.\n";
        return;
    }
    // One cent of quote error, expressed in vol points, is the form a trader
    // reads this in.
    const double vol_points_per_cent = 100.0 * 0.01 / std::abs(iv.vega);
    std::cout << "  Sensitivity : a 0.01 quote error moves the implied vol by "
              << vol_points_per_cent << " vol points\n";
}

[[nodiscard]] int run(const Config& cfg) {
    opt::MarketData market{cfg.spot, cfg.rate, cfg.dividend, cfg.volatility};

    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);

    opt::IvResult iv{};
    if (cfg.solve_iv) {
        iv = solve_implied_volatility(cfg, market);
        if (!iv.converged()) {
            std::cerr << "error: no implied volatility (" << opt::to_string(iv.status)
                      << "): " << iv_diagnosis(iv.status) << '\n';
            return 1;
        }
        // Everything downstream -- the Greeks above all -- is reported at the
        // volatility that was recovered, not at the zero it started from.
        market.volatility = iv.volatility;
    }

    const opt::Greeks greeks = opt::bs_greeks(market, cfg.strike, cfg.expiry, cfg.type);

    std::cout << "Inputs\n"
              << "  Method      : " << method_name(cfg.method)
              << (cfg.solve_iv ? "  [solving for volatility]" : "") << '\n'
              << "  Type        : " << (cfg.type == opt::OptionType::Call ? "Call" : "Put") << " ("
              << (cfg.exercise == opt::Exercise::European ? "European" : "American") << ")\n"
              << "  Spot        : " << cfg.spot << '\n'
              << "  Strike      : " << cfg.strike << '\n'
              << "  Rate        : " << cfg.rate << '\n'
              << "  Dividend    : " << cfg.dividend << '\n';
    if (cfg.solve_iv) {
        std::cout << "  Market price: " << cfg.target_price << '\n';
    } else {
        std::cout << "  Volatility  : " << cfg.volatility << '\n';
    }
    std::cout << "  Expiry      : " << cfg.expiry << " years\n\n";

    if (cfg.solve_iv) {
        print_implied_volatility(iv);
        if (cfg.method == Method::Tree) {
            std::cout << "  (" << cfg.steps << " steps)\n";
        }
        std::cout << "\nGreeks (Black-Scholes analytical, at the implied volatility):\n"
                  << "  Delta       : " << greeks.delta << '\n'
                  << "  Gamma       : " << greeks.gamma << '\n'
                  << "  Vega        : " << greeks.vega << "   (per 1.00 vol)\n"
                  << "  Theta       : " << greeks.theta << "   (per year)\n"
                  << "  Rho         : " << greeks.rho << "   (per 1.00 rate)\n";
        return 0;
    }

    std::cout << "Price         : ";
    if (cfg.method == Method::BlackScholes) {
        std::cout << opt::black_scholes_price(market, cfg.strike, cfg.expiry, cfg.type) << '\n';
    } else if (cfg.method == Method::Tree) {
        const std::unique_ptr<opt::Option> option = make_option(cfg);
        std::cout << opt::binomial_price(*option, market, cfg.steps) << '\n'
                  << "  (" << cfg.steps << " steps)\n";
    } else {
        // Monte Carlo prices a European-style (terminal) payoff.
        const opt::EuropeanCall call{cfg.strike, cfg.expiry};
        const opt::EuropeanPut put{cfg.strike, cfg.expiry};
        const opt::Option& option =
            cfg.type == opt::OptionType::Call ? static_cast<const opt::Option&>(call) : put;
        opt::McSettings settings;
        settings.num_paths = cfg.paths;
        settings.seed = cfg.seed;
        settings.variance_reduction = cfg.variates;
        const opt::McResult mc = opt::monte_carlo_price(option, market, settings);
        std::cout << mc.price << '\n'
                  << "  Std error   : " << mc.std_error << '\n'
                  << "  95% CI      : [" << mc.ci_low << ", " << mc.ci_high << "]\n"
                  << "  Paths       : " << mc.samples
                  << "  (variates: " << variates_name(cfg.variates) << ")\n";
    }

    std::cout << "\nGreeks (Black-Scholes analytical):\n"
              << "  Delta       : " << greeks.delta << '\n'
              << "  Gamma       : " << greeks.gamma << '\n'
              << "  Vega        : " << greeks.vega << "   (per 1.00 vol)\n"
              << "  Theta       : " << greeks.theta << "   (per year)\n"
              << "  Rho         : " << greeks.rho << "   (per 1.00 rate)\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Config cfg;
        if (!parse_args(argc, argv, cfg)) {
            return 0;  // --help
        }
        return run(cfg);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n\n";
        print_usage(std::cerr);
        return 1;
    }
}
