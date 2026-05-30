# options-pricing-engine

**Price European & American options three independent ways — and make them agree.**

A multi-method options pricing library in modern C++20: closed-form
**Black-Scholes-Merton**, a **Cox-Ross-Rubinstein binomial tree**, and
**Monte Carlo** simulation, with analytical *and* finite-difference Greeks, a
vectorised pricing grid, a rigorous GoogleTest suite, and a throughput
benchmark. Every price is cross-validated across all three code paths, so no
single method is ever trusted on its own.

![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.20%2B-064F8C?logo=cmake&logoColor=white)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)



## Demo

Pricing the benchmark European call (`S=100, K=100, r=5%, σ=20%, T=1y`) from the
command line — the analytical price, the lattice that converges to it, and the
Monte Carlo estimate whose 95% confidence interval brackets both:

```console
$ ./build/bin/price_cli --spot 100 --strike 100 --rate 0.05 --vol 0.2 --expiry 1 --type call --method bs
Inputs
  Method      : Black-Scholes (analytical)
  Type        : Call (European)
  Spot        : 100.000000
  Strike      : 100.000000
  Rate        : 0.050000
  Dividend    : 0.000000
  Volatility  : 0.200000
  Expiry      : 1.000000 years

Price         : 10.450584

Greeks (Black-Scholes analytical):
  Delta       : 0.636831
  Gamma       : 0.018762
  Vega        : 37.524035   (per 1.00 vol)
  Theta       : -6.414028   (per year)
  Rho         : 53.232482   (per 1.00 rate)

$ ./build/bin/price_cli ... --method mc --paths 1000000
Price         : 10.454014
  Std error   : 0.002740
  95% CI      : [10.448645, 10.459384]
  Paths       : 500000  (variates: antithetic+control)
```

The full test suite and the vectorised-grid benchmark both run with one command:

```console
$ ctest --test-dir build
100% tests passed, 0 tests failed out of 36
Total Test time (real) =   0.14 sec

$ ./build/bin/opt_bench
grid            scalar (ms)     eigen (ms)    speedup     scalar opt/s      eigen opt/s
--------------------------------------------------------------------------------
  100x100             0.252          0.092      2.73x        3.971e+07        1.083e+08
  250x250             1.422          0.445      3.20x        4.397e+07        1.406e+08
  500x500             5.416          1.554      3.48x        4.616e+07        1.608e+08
 1000x1000           21.167          5.882      3.60x        4.724e+07        1.700e+08
```

> **Tip:** to make the repo feel even more alive, record the session above as an
> animated GIF (e.g. with [`asciinema`](https://asciinema.org) +
> [`agg`](https://github.com/asciinema/agg), or [`vhs`](https://github.com/charmbracelet/vhs))
> and drop it in as `docs/demo.gif`, then embed it here:
> `![demo](docs/demo.gif)`.

---

## Overview

Pricing the same instrument by analytics, by a lattice, and by simulation
exercises three quite different parts of a quant developer's toolkit —
special-function accuracy, backward induction under tight memory, and
variance-reduced stochastic estimation — and forces them to agree. The library
is deliberately written to read like production C++ (RAII, `const`-correctness,
`[[nodiscard]]`, no global state, no raw `new`/`delete`, warnings-as-errors)
rather than a notebook ported to a `.cpp` file.

If you want one sentence: **it is a portfolio-grade demonstration that the math,
the engineering, and the numerical agreement all hold up at the same time.**

---

## Features

| Method | European | American | Greeks | Notes |
|--------|:--------:|:--------:|:------:|-------|
| Black-Scholes-Merton (analytical) | ✅ | — | analytical + finite-difference | dividend yield supported |
| CRR binomial tree | ✅ | ✅ | (price) | `O(N)` memory backward induction |
| Monte Carlo (GBM terminal) | ✅ | — | (price) | antithetic + control variates, 95% CI |
| Vectorised grid | ✅ | — | — | Eigen/SIMD, ~3.6× over a scalar loop |

- **Three cross-validated engines** — analytical, lattice, and simulation agree
  to within their expected error on every test case.
- **Full Greeks suite** — delta, gamma, vega, theta, rho, computed both
  closed-form and by central finite difference, agreeing to `1e-4` across a
  parameter sweep.
- **Reproducible Monte Carlo** — deterministically-seeded `std::mt19937_64`
  returning the estimate, standard error, and 95% confidence interval.
- **Variance reduction** — antithetic variates and an asset-price control
  variate cut MC variance ~28× on the worked example.
- **Vectorised pricing grid** — an Eigen/SIMD CDF prices a whole surface without
  a scalar `libm` call, ~3.6× faster than a per-cell loop.
- **Engineered like production** — `-Wall -Wextra -Wpedantic -Werror`,
  dependencies fetched (not vendored), and a one-command test + benchmark flow.

---

## Prerequisites & Dependencies

| Requirement | Version | Notes |
|-------------|---------|-------|
| C++ compiler | C++20 | GCC 11+, Clang 13+, or MSVC 19.30+ |
| CMake | 3.20+ | |
| [Eigen](https://eigen.tuxfamily.org) | 3.4 | fetched automatically — nothing to install |
| [GoogleTest](https://github.com/google/googletest) | 1.14 | fetched automatically — nothing to install |

Eigen and GoogleTest are pulled in via CMake `FetchContent` on the first
configure, so the only thing you need system-wide is a compiler and CMake.
**Network access is required on the first configure** to fetch them.

---

## Getting Started

### 1. Clone

```sh
git clone https://github.com/leeyawnnn/options-pricing-engine.git
cd options-pricing-engine
```

### 2. Build (Linux / macOS)

```sh
cmake -B build           # first configure fetches Eigen + GoogleTest
cmake --build build -j
```

### Build (Windows / MSVC)

```bat
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

The build is warnings-as-errors (`-Wall -Wextra -Wpedantic -Werror`, or
`/W4 /WX /permissive-` on MSVC) for the project's own targets; third-party
headers are treated as system headers and excluded from that policy.

**Targets produced:**

| Target      | Kind            | Description                       |
|-------------|-----------------|-----------------------------------|
| `opt_lib`   | static library  | the pricing engine                |
| `opt_tests` | test executable | GoogleTest unit tests             |
| `price_cli` | executable      | command-line pricing demo         |
| `opt_bench` | executable      | loop-vs-vectorised grid benchmark |

Binaries land in `build/bin/`, the static library in `build/lib/`.

### 3. Run the tests

```sh
ctest --test-dir build --output-on-failure   # 36 tests, ~0.1s
```

---

## Usage

### Worked example — pricing a call three ways

The same European call (`S=100, K=100, r=5%, σ=20%, T=1y`) priced by each
method. The analytical value is `10.4506`; the tree converges to it and the
Monte Carlo confidence interval brackets it.

```sh
./build/bin/price_cli --spot 100 --strike 100 --rate 0.05 --vol 0.2 --expiry 1 --type call --method bs
./build/bin/price_cli --spot 100 --strike 100 --rate 0.05 --vol 0.2 --expiry 1 --type call --method tree --steps 5000
./build/bin/price_cli --spot 100 --strike 100 --rate 0.05 --vol 0.2 --expiry 1 --type call --method mc --paths 1000000
```

| Method | Price | Detail |
|--------|-------|--------|
| Black-Scholes | **10.450584** | Δ 0.6368, Γ 0.0188, vega 37.52, θ −6.41/yr, ρ 53.23 |
| Binomial tree (5000 steps) | **10.450184** | error ≈ 4e-4, consistent with `O(1/N)` |
| Monte Carlo (10⁶ paths) | **10.454014** | std err 0.00274, 95% CI **[10.4486, 10.4594]** ✓ brackets BS |

### CLI flags

```
--spot --strike --rate --vol --expiry        (required)
--dividend --type call|put --exercise european|american
--method bs|tree|mc --steps N --paths N --seed N
--variates none|antithetic|control|both
```

### Library use

`opt_lib` is a plain static library exposing the public headers under
`include/opt/`. After building, link against the `opt::lib` target from your own
CMake project (e.g. via `add_subdirectory` or `FetchContent`) and include the
header for the method you need:

```cpp
#include <opt/black_scholes.hpp>
#include <opt/market_data.hpp>

// One MarketData can price a whole book; expiry lives on the contract.
const opt::MarketData market{.spot = 100.0, .rate = 0.05,
                             .dividend = 0.0, .volatility = 0.20};

// European call: K=100, T=1y  ->  price ≈ 10.4506
double price = opt::black_scholes_call(market, /*strike=*/100.0, /*expiry=*/1.0);
```

---

## Performance

Vectorised `price_grid` (Eigen/SIMD) vs a scalar per-cell loop, Apple clang,
`-O3`, best of 5 runs:

| Grid        | Scalar   | Eigen   | Speedup | Eigen throughput  |
|-------------|---------:|--------:|:-------:|------------------:|
| 100 × 100   | 0.25 ms  | 0.09 ms | 2.7×    | ~108 M options/s  |
| 250 × 250   | 1.42 ms  | 0.45 ms | 3.2×    | ~141 M options/s  |
| 500 × 500   | 5.42 ms  | 1.55 ms | 3.5×    | ~161 M options/s  |
| 1000 × 1000 | 21.17 ms | 5.88 ms | 3.6×    | ~170 M options/s  |

The vectorised grid uses a SIMD-friendly Abramowitz-Stegun CDF (`|err| < 7.5e-8`
in Φ) so the whole grid runs without a scalar `libm` call; the exact `erfc`-based
engine remains the single-price reference. Reproduce with `./build/bin/opt_bench`.

---

## Math Reference

Notation: spot `S`, strike `K`, risk-free rate `r`, dividend yield `q`,
volatility `σ`, time to expiry `T`, standard normal pdf/cdf `φ`/`Φ`.

| Quantity | Formula | Function |
|----------|---------|----------|
| `d₁`, `d₂` | `d₁ = [ln(S/K) + (r−q+½σ²)T] / (σ√T)`, `d₂ = d₁ − σ√T` | internal |
| Standard normal CDF | `Φ(x) = ½·erfc(−x/√2)` | `cumulative_normal` |
| Call | `C = S e^{−qT} Φ(d₁) − K e^{−rT} Φ(d₂)` | `black_scholes_call` |
| Put | `P = K e^{−rT} Φ(−d₂) − S e^{−qT} Φ(−d₁)` | `black_scholes_put` |
| Put-call parity | `C − P = S e^{−qT} − K e^{−rT}` | `put_call_parity_residual` |
| Delta | `e^{−qT} Φ(d₁)` (call) | `bs_delta` |
| Gamma | `e^{−qT} φ(d₁) / (S σ√T)` | `bs_gamma` |
| Vega | `S e^{−qT} φ(d₁) √T` | `bs_vega` |
| Theta | `∂V/∂t` (per year) | `bs_theta` |
| Rho | `K T e^{−rT} Φ(d₂)` (call) | `bs_rho` |
| CRR lattice | `u = e^{σ√Δt}`, `d = 1/u`, `p = (e^{(r−q)Δt} − d)/(u − d)` | `binomial_price` |
| GBM terminal | `S_T = S₀ exp((r−q−½σ²)T + σ√T·Z)`, `Z∼N(0,1)` | `monte_carlo_price` |

Full derivations are in the Doxygen comments on each public header.

---

## Project Layout

```
include/opt/   public headers (market data, options, all pricing methods)
src/           library implementation (+ bs_internal.hpp, a private detail header)
tests/         GoogleTest unit tests, one file per component
benchmarks/    hand-rolled timing harness (options/second)
apps/          the price_cli demo
```

---

## Design Notes

- **CDF via `erfc`, not a low-order rational.** A rational CDF approximation
  accurate enough for *prices* can still be too rough for *Greeks*, because the
  Greeks differentiate it. `Φ(x) = ½·erfc(−x/√2)` is accurate to ~1e-16, which
  keeps analytical and finite-difference Greeks consistent.
- **`O(N)` tree.** A single `std::vector<double>` is rolled back in place rather
  than materialising the full `O(N²)` lattice.
- **Variance reduction.** Antithetic variates and an asset-price control variate
  (with its analytically-known forward mean) cut the Monte Carlo variance ~28×
  on the worked example.
- **Exercise lives on the option.** The `Option` hierarchy carries the exercise
  style, so the binomial pricer branches on the contract rather than on a flag.

---

## Roadmap

- **Stochastic volatility** (Heston) and **local volatility** surfaces.
- **Jump-diffusion** (Merton, Kou).
- **Exotics:** barrier, Asian, and lookback options (the Monte Carlo engine is
  structured to grow full-path simulation).
- **American Greeks** by finite differencing the lattice, and **Longstaff-
  Schwartz** least-squares Monte Carlo for early exercise.
- **Implied volatility** solving (Newton / Brent).

---

## Contributing

Contributions, bug reports, and feature ideas are welcome.

1. **Open an issue** describing the change before large PRs, so we can agree on
   approach.
2. **Fork & branch** from `main` (`git checkout -b feature/my-change`).
3. **Keep the build green** — the project is warnings-as-errors and every new
   pricing path is expected to ship with GoogleTest coverage that
   cross-validates it against an existing method:

   ```sh
   cmake --build build -j && ctest --test-dir build --output-on-failure
   ```

4. **Match the house style** — run `clang-format` (the repo ships a
   `.clang-format`) and keep the existing conventions: `const`-correctness,
   `[[nodiscard]]`, no global state, no raw `new`/`delete`.
5. **Open a PR** with a clear description and any relevant before/after numbers
   (prices, errors, benchmark timings).

Good first contributions are listed in the [Roadmap](#roadmap) above — the
implied-volatility solver and the barrier/Asian exotics are the most
self-contained starting points.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
