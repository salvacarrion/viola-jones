import os
import time
import tempfile
import argparse
import numpy as np
from adaboost import AdaBoost

def main():
    parser = argparse.ArgumentParser(description="Benchmark AdaBoost precomputed sort index (int32 vs uint16 vs no-precompute)")
    parser.add_argument("--n-features", type=int, default=2000, help="Number of features for the benchmark")
    parser.add_argument("--n-samples", type=int, default=49200, help="Number of training samples")
    parser.add_argument("--n-rounds", type=int, default=5, help="Number of boosting rounds")
    parser.add_argument("--chunk-size", type=int, default=500, help="Feature chunk size")
    args = parser.parse_args()

    n_features = args.n_features
    n_samples = args.n_samples
    n_rounds = args.n_rounds
    feature_chunk = args.chunk_size

    print("=" * 60)
    print("AdaBoost Sorting Index Precomputation Benchmark")
    print(f"Features:  {n_features:,}")
    print(f"Samples:   {n_samples:,}")
    print(f"Rounds:    {n_rounds}")
    print(f"Dtype:     {'uint16' if n_samples <= 65535 else 'int32'} (Auto-selected)")
    print("=" * 60)

    # Generate synthetic training data
    print("Generating synthetic data...")
    rng = np.random.default_rng(42)
    # Generating standard normal features to simulate float32 feature matrix
    X = rng.standard_normal((n_features, n_samples), dtype=np.float32)
    y = rng.choice([0.0, 1.0], size=n_samples).astype(np.float32)
    features = [None] * n_features

    # 1. Baseline: No precompute
    print("\n--- Running Baseline (No Precompute) ---")
    ab_no = AdaBoost(n_estimators=n_rounds)
    start = time.time()
    ab_no.train(X, y, features, feature_chunk=feature_chunk, precompute_sort_index=False)
    t_no_total = time.time() - start
    t_no_per_round = t_no_total / n_rounds
    print(f"Baseline total time: {t_no_total:.2f}s ({t_no_per_round:.4f}s per round)")

    # 2. Precompute: uint16 (or int32 if samples > 65535)
    print("\n--- Running Precompute (Auto-selected dtype) ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        ab_pre = AdaBoost(n_estimators=n_rounds)
        start = time.time()
        ab_pre.train(X, y, features, feature_chunk=feature_chunk,
                     precompute_sort_index=True, sort_index_dir=tmpdir)
        t_pre_total = time.time() - start
        
        # We can extract the precomputation time printout context
        # Let's measure the precompute time directly to separate it
        # from the boosting phase.
        # We do a direct precomputation to get precise timing.
        idx_dtype = np.uint16 if n_samples <= 65535 else np.int32
        path_test = os.path.join(tmpdir, "_benchmark_sort_idx.npy")
        t_write_start = time.time()
        _wmap = np.memmap(path_test, dtype=idx_dtype, mode='w+', shape=(n_features, n_samples))
        for f0 in range(0, n_features, feature_chunk):
            f1 = min(f0 + feature_chunk, n_features)
            _wmap[f0:f1] = np.argsort(X[f0:f1], axis=1, kind='quicksort').astype(idx_dtype)
        _wmap.flush()
        del _wmap
        t_write_pre = time.time() - t_write_start
        
        t_boost_phase = t_pre_total - t_write_pre
        t_boost_per_round = t_boost_phase / n_rounds
        print(f"Precomputed total time:      {t_pre_total:.2f}s")
        print(f"  - Precompute (write) phase: {t_write_pre:.2f}s")
        print(f"  - Boosting rounds phase:    {t_boost_phase:.2f}s ({t_boost_per_round:.4f}s per round)")

    # 3. Direct Type Comparison (int32 vs uint16)
    print("\n--- Direct Disk-backed Precomputation Benchmark (int32 vs uint16) ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Benchmark int32
        path_32 = os.path.join(tmpdir, "idx_32.npy")
        t_w32_start = time.time()
        wmap_32 = np.memmap(path_32, dtype=np.int32, mode='w+', shape=(n_features, n_samples))
        for f0 in range(0, n_features, feature_chunk):
            f1 = min(f0 + feature_chunk, n_features)
            wmap_32[f0:f1] = np.argsort(X[f0:f1], axis=1, kind='quicksort').astype(np.int32)
        wmap_32.flush()
        del wmap_32
        t_w32 = time.time() - t_w32_start
        size_32 = os.path.getsize(path_32)

        t_r32_start = time.time()
        rmap_32 = np.memmap(path_32, dtype=np.int32, mode='r', shape=(n_features, n_samples))
        # Simulate access pattern: read 5 chunks of features
        for f0 in range(0, min(feature_chunk * 5, n_features), feature_chunk):
            f1 = min(f0 + feature_chunk, n_features)
            _ = rmap_32[f0:f1]
        t_r32 = time.time() - t_r32_start

        # Benchmark uint16
        path_16 = os.path.join(tmpdir, "idx_16.npy")
        t_w16_start = time.time()
        wmap_16 = np.memmap(path_16, dtype=np.uint16, mode='w+', shape=(n_features, n_samples))
        for f0 in range(0, n_features, feature_chunk):
            f1 = min(f0 + feature_chunk, n_features)
            wmap_16[f0:f1] = np.argsort(X[f0:f1], axis=1, kind='quicksort').astype(np.uint16)
        wmap_16.flush()
        del wmap_16
        t_w16 = time.time() - t_w16_start
        size_16 = os.path.getsize(path_16)

        t_r16_start = time.time()
        rmap_16 = np.memmap(path_16, dtype=np.uint16, mode='r', shape=(n_features, n_samples))
        for f0 in range(0, min(feature_chunk * 5, n_features), feature_chunk):
            f1 = min(f0 + feature_chunk, n_features)
            _ = rmap_16[f0:f1]
        t_r16 = time.time() - t_r16_start

        print(f"int32 precomputation:  Write={t_w32:.2f}s, Read (5 chunks)={t_r32:.4f}s, Size={size_32 / 1e6:.1f} MB")
        print(f"uint16 precomputation: Write={t_w16:.2f}s, Read (5 chunks)={t_r16:.4f}s, Size={size_16 / 1e6:.1f} MB")
        print(f"IO/Disk Savings:       {100 * (1 - size_16/size_32):.1f}% reduction in file size")
        print(f"Write Speedup:         {t_w32 / t_w16:.2f}x faster write")

    # 4. Extrapolation
    print("\n" + "=" * 60)
    print("ESTRAPOLATION TO FULL SCALE (60,153 features, 49,200 samples, 1,200 WCs)")
    print("=" * 60)
    
    full_features = 60153
    full_rounds = 1200
    
    # Calculate scale factor for features and rounds
    feature_factor = full_features / n_features
    
    # Baseline: No precompute (argsort every round)
    # Total time = n_rounds * round_time
    est_total_no_precompute = t_no_per_round * feature_factor * full_rounds
    
    # Precomputed: uint16
    # Total time = precompute_time + n_rounds * round_time
    est_precompute_write_16 = t_w16 * feature_factor
    est_boosting_rounds_precomputed = t_boost_per_round * feature_factor * full_rounds
    est_total_precomputed_16 = est_precompute_write_16 + est_boosting_rounds_precomputed

    # Precomputed: int32
    est_precompute_write_32 = t_w32 * feature_factor
    est_total_precomputed_32 = est_precompute_write_32 + est_boosting_rounds_precomputed

    size_full_32 = full_features * n_samples * 4 / 1e9
    size_full_16 = full_features * n_samples * 2 / 1e9

    print(f"Estimated baseline training (No Precompute): {est_total_no_precompute / 3600:.2f} hours")
    print(f"Estimated training with precompute (int32):  {est_total_precomputed_32 / 3600:.2f} hours")
    print(f"  - Precompute phase:                        {est_precompute_write_32:.1f} seconds")
    print(f"  - Boosting phase:                          {est_boosting_rounds_precomputed / 3600:.2f} hours")
    print(f"  - Disk footprint:                          {size_full_32:.2f} GB")
    print()
    print(f"Estimated training with precompute (uint16): {est_total_precomputed_16 / 3600:.2f} hours")
    print(f"  - Precompute phase:                        {est_precompute_write_16:.1f} seconds")
    print(f"  - Boosting phase:                          {est_boosting_rounds_precomputed / 3600:.2f} hours")
    print(f"  - Disk footprint:                          {size_full_16:.2f} GB")
    print()
    print(f"Estimated overall speedup:                   {est_total_no_precompute / est_total_precomputed_16:.2f}x")
    print(f"Disk & RAM page-cache footprint reduction:   {(size_full_32 - size_full_16):.2f} GB (50% reduction)")

if __name__ == "__main__":
    main()
