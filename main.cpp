#include <benchmark/benchmark.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <random>

constexpr unsigned int N = 128;
constexpr unsigned int NUM_ITERATIONS = 2500;

static void matmul() {
    double a[N][N];
    double b[N][N];
    double c[N][N];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (unsigned int i = 0; i < N; i++) {
        for (unsigned int j = 0; j < N; j++) {
            a[i][j] = dist(gen);
            b[i][j] = dist(gen);
            c[i][j] = 0.0;
        }
    }

    for (unsigned int i = 0; i < N; i++) {
        for (unsigned int j = 0; j < N; j++) {
            for (unsigned int k = 0; k < N; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

static void BM_matmul(benchmark::State& state) {
    auto start = std::chrono::high_resolution_clock::now();
    for (auto _ : state) {
        matmul();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start) / state.threads();
    state.counters["TotalTimeSec"] = benchmark::Counter(elapsed_seconds.count(), benchmark::Counter::kDefaults);
    state.counters["IterationsPerSec"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}

BENCHMARK(BM_matmul)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(NUM_ITERATIONS)
    ->ThreadRange(1, std::thread::hardware_concurrency());

BENCHMARK_MAIN();
