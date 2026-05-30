// price_cli -- command-line front-end to the pricing engine.
//
// Example:
//   price_cli --spot 100 --strike 105 --rate 0.05 --vol 0.2 --expiry 1.0 \
//             --type call --method bs
//
// Prints the price (by the chosen method), the analytical Black-Scholes
// Greeks, and -- for Monte Carlo -- the standard error and 95% confidence
// interval. The argument parser is hand-rolled; there are no external
// dependencies beyond the engine itself.
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
    int steps = 1000;                                            // tree
    std::size_t paths = 100'000;                                 // Monte Carlo
    std::uint64_t seed = 0x5DEECE66DULL;                         // Monte Carlo
    opt::VarianceReduction variates = opt::VarianceReduction::AntitheticAndControl;
};

void print_usage(std::ostream& os) {
    os << "Usage: price_cli [options]\n\n"
          "Required:\n"
          "  --spot <S>        underlying spot price (> 0)\n"
          "  --strike <K>      strike price (> 0)\n"
          "  --rate <r>        continuously-compounded risk-free rate\n"
          "  --vol <sigma>     annualised volatility (>= 0)\n"
          "  --expiry <T>      time to expiry in years (>= 0)\n\n"
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
    if (v == "call") return opt::OptionType::Call;
    if (v == "put") return opt::OptionType::Put;
    throw std::runtime_error("--type must be 'call' or 'put', got '" + v + "'");
}

[[nodiscard]] opt::Exercise parse_exercise(const std::string& v) {
    if (v == "european") return opt::Exercise::European;
    if (v == "american") return opt::Exercise::American;
    throw std::runtime_error("--exercise must be 'european' or 'american', got '" + v + "'");
}

[[nodiscard]] Method parse_method(const std::string& v) {
    if (v == "bs") return Method::BlackScholes;
    if (v == "tree") return Method::Tree;
    if (v == "mc") return Method::MonteCarlo;
    throw std::runtime_error("--method must be 'bs', 'tree' or 'mc', got '" + v + "'");
}

[[nodiscard]] opt::VarianceReduction parse_variates(const std::string& v) {
    if (v == "none") return opt::VarianceReduction::None;
    if (v == "antithetic") return opt::VarianceReduction::Antithetic;
    if (v == "control") return opt::VarianceReduction::ControlVariate;
    if (v == "both") return opt::VarianceReduction::AntitheticAndControl;
    throw std::runtime_error("--variates must be none|antithetic|control|both, got '" + v + "'");
}

// Parses argv into a Config. Returns false (after printing usage) when --help
// was requested; throws std::runtime_error on a malformed command line.
[[nodiscard]] bool parse_args(int argc, char** argv, Config& cfg) {
    bool have_spot = false, have_strike = false, have_rate = false;
    bool have_vol = false, have_expiry = false;

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

        if (flag == "--spot") { cfg.spot = to_double(flag, value()); have_spot = true; }
        else if (flag == "--strike") { cfg.strike = to_double(flag, value()); have_strike = true; }
        else if (flag == "--rate") { cfg.rate = to_double(flag, value()); have_rate = true; }
        else if (flag == "--vol") { cfg.volatility = to_double(flag, value()); have_vol = true; }
        else if (flag == "--expiry") { cfg.expiry = to_double(flag, value()); have_expiry = true; }
        else if (flag == "--dividend") { cfg.dividend = to_double(flag, value()); }
        else if (flag == "--type") { cfg.type = parse_type(value()); }
        else if (flag == "--exercise") { cfg.exercise = parse_exercise(value()); }
        else if (flag == "--method") { cfg.method = parse_method(value()); }
        else if (flag == "--steps") { cfg.steps = static_cast<int>(to_int(flag, value())); }
        else if (flag == "--paths") { cfg.paths = static_cast<std::size_t>(to_int(flag, value())); }
        else if (flag == "--seed") { cfg.seed = static_cast<std::uint64_t>(to_int(flag, value())); }
        else if (flag == "--variates") { cfg.variates = parse_variates(value()); }
        else { throw std::runtime_error("unknown flag: " + flag); }
    }

    if (!(have_spot && have_strike && have_rate && have_vol && have_expiry)) {
        throw std::runtime_error("missing required flag(s); need --spot --strike --rate --vol --expiry");
    }
    if (cfg.spot <= 0.0) throw std::runtime_error("--spot must be > 0");
    if (cfg.strike <= 0.0) throw std::runtime_error("--strike must be > 0");
    if (cfg.volatility < 0.0) throw std::runtime_error("--vol must be >= 0");
    if (cfg.expiry < 0.0) throw std::runtime_error("--expiry must be >= 0");
    if (cfg.method == Method::Tree && cfg.steps < 1) throw std::runtime_error("--steps must be >= 1");
    return true;
}

[[nodiscard]] std::unique_ptr<opt::Option> make_option(const Config& cfg) {
    const bool call = cfg.type == opt::OptionType::Call;
    if (cfg.exercise == opt::Exercise::American) {
        if (call) return std::make_unique<opt::AmericanCall>(cfg.strike, cfg.expiry);
        return std::make_unique<opt::AmericanPut>(cfg.strike, cfg.expiry);
    }
    if (call) return std::make_unique<opt::EuropeanCall>(cfg.strike, cfg.expiry);
    return std::make_unique<opt::EuropeanPut>(cfg.strike, cfg.expiry);
}

const char* method_name(Method m) {
    switch (m) {
        case Method::BlackScholes: return "Black-Scholes (analytical)";
        case Method::Tree: return "CRR binomial tree";
        case Method::MonteCarlo: return "Monte Carlo";
    }
    return "unknown";
}

const char* variates_name(opt::VarianceReduction v) {
    switch (v) {
        case opt::VarianceReduction::None: return "none";
        case opt::VarianceReduction::Antithetic: return "antithetic";
        case opt::VarianceReduction::ControlVariate: return "control";
        case opt::VarianceReduction::AntitheticAndControl: return "antithetic+control";
    }
    return "unknown";
}

void run(const Config& cfg) {
    const opt::MarketData market{cfg.spot, cfg.rate, cfg.dividend, cfg.volatility};
    const opt::Greeks greeks = opt::bs_greeks(market, cfg.strike, cfg.expiry, cfg.type);

    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);

    std::cout << "Inputs\n"
              << "  Method      : " << method_name(cfg.method) << '\n'
              << "  Type        : " << (cfg.type == opt::OptionType::Call ? "Call" : "Put") << " ("
              << (cfg.exercise == opt::Exercise::European ? "European" : "American") << ")\n"
              << "  Spot        : " << cfg.spot << '\n'
              << "  Strike      : " << cfg.strike << '\n'
              << "  Rate        : " << cfg.rate << '\n'
              << "  Dividend    : " << cfg.dividend << '\n'
              << "  Volatility  : " << cfg.volatility << '\n'
              << "  Expiry      : " << cfg.expiry << " years\n\n";

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
                  << "  Paths       : " << mc.samples << "  (variates: "
                  << variates_name(cfg.variates) << ")\n";
    }

    std::cout << "\nGreeks (Black-Scholes analytical):\n"
              << "  Delta       : " << greeks.delta << '\n'
              << "  Gamma       : " << greeks.gamma << '\n'
              << "  Vega        : " << greeks.vega << "   (per 1.00 vol)\n"
              << "  Theta       : " << greeks.theta << "   (per year)\n"
              << "  Rho         : " << greeks.rho << "   (per 1.00 rate)\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Config cfg;
        if (!parse_args(argc, argv, cfg)) {
            return 0;  // --help
        }
        run(cfg);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n\n";
        print_usage(std::cerr);
        return 1;
    }
    return 0;
}
