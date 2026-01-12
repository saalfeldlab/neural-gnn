import numpy as np
import math
from dataclasses import dataclass
from tqdm import trange
from itertools import product
from math import log
from tqdm import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt

def analyze_ising_model(x_list, delta_t, log_dir, logger, edges):
    """
    Perform comprehensive Ising model analysis including energy distribution,
    coupling analysis, information ratio estimation, and triplet KL analysis.
    
    Args:
        x_list: List of input data arrays
        log_dir: Directory for saving results
        logger: Logger object for recording results
        edges: Edge list for connectivity-aware sampling
    
    Returns:
        dict: Dictionary containing all analysis results
    """
    
    # binarize at per-neuron mean -> {-1,+1}
    voltage = x_list[0][:, :, 3]
    mean_v = voltage.mean(axis=0)
    s = np.where(voltage > mean_v, 1, -1).astype(np.int8)
    
    # Random sampling (baseline)
    results_random = compute_entropy_analysis(s, delta_t, log_dir, logger, n_subsets=1000, N=10)
    
    # Connectivity-aware sampling
    results_connected = analyze_higher_order_correlations(s, edges, delta_t, log_dir, logger, n_subsets=1000, N=10)
    
    # Calculate enrichment metrics
    enrichment_metrics = calculate_enrichment_metrics(results_random, results_connected, s, edges, logger)
    
    # Combine all results
    results_dict = {
        'random_sampling': results_random,
        'connected_sampling': results_connected,
        'enrichment': enrichment_metrics
    }
    
    return results_dict


def calculate_enrichment_metrics(results_random, results_connected, s, edges, logger):
    """
    Calculate and log enrichment metrics comparing random vs connected sampling
    """
    
    # Basic enrichment ratios (using medians)
    enrichment_I_N = results_connected['I_N_median'] / (results_random['I_N_median'] + 1e-10)
    enrichment_I_2 = results_connected['I_2_median'] / (results_random['I_2_median'] + 1e-10)
    enrichment_I_HOC = results_connected['I_HOC_median'] / (results_random['I_HOC_median'] + 1e-10)
    
    # Extract C_3 for random sampling (need to calculate if not present)
    if 'C_3_mean' not in results_random:
        # Calculate C_3 for a subset of random samples for comparison
        C_3_random = calculate_C3_for_random_samples(s, n_samples=100)
        results_random['C_3_mean'] = np.mean(C_3_random)
        results_random['C_3_std'] = np.std(C_3_random)
    
    enrichment_C3 = results_connected['C_3_mean'] / (results_random['C_3_mean'] + 1e-10)
    
    # Statistical comparison using Mann-Whitney U test
    from scipy.stats import mannwhitneyu
    
    # Compare distributions of I_N values
    I_N_random = results_random['I_N']
    I_N_connected = results_connected['I_N']
    
    if len(I_N_random) > 0 and len(I_N_connected) > 0:
        statistic, pvalue = mannwhitneyu(I_N_connected, I_N_random, alternative='greater')
        
        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(I_N_connected) - np.mean(I_N_random)
        pooled_std = np.sqrt((np.var(I_N_connected) + np.var(I_N_random)) / 2)
        cohens_d = mean_diff / (pooled_std + 1e-10)
    else:
        pvalue = np.nan
        cohens_d = np.nan
    
    # Percentage of connected subgraphs with higher I_N than median random
    median_random_I_N = np.median(I_N_random)
    pct_connected_above_median = np.mean(I_N_connected > median_random_I_N) * 100
    
    # Calculate edge density metrics
    edge_density_random, edge_density_connected = calculate_edge_densities(s, edges)
    
    # Calculate correlation decay with distance
    correlation_decay = analyze_correlation_decay(s, edges)
    
    # Calculate signal-to-noise ratio
    snr = calculate_snr(s, edges)
    
    # Log all enrichment metrics
    logger.info("\n" + "="*60)
    logger.info("=== ENRICHMENT ANALYSIS ===")
    logger.info("="*60)
    
    logger.info(f"Enrichment I_N:    {enrichment_I_N:.3f}x")
    logger.info(f"Enrichment I_2:    {enrichment_I_2:.3f}x")
    logger.info(f"Enrichment I_HOC:  {enrichment_I_HOC:.3f}x")
    logger.info(f"Enrichment C_3:    {enrichment_C3:.3f}x")
    
    logger.info("\nStatistical comparison (Mann-Whitney U):")
    logger.info(f"  p-value: {pvalue:.4e}")
    logger.info(f"  Cohen's d: {cohens_d:.3f}")
    logger.info(f"  Connected > median(random): {pct_connected_above_median:.1f}%")
    
    logger.info("\nConnectivity statistics:")
    logger.info(f"  Edge density random:    {edge_density_random:.4f}")
    logger.info(f"  Edge density connected: {edge_density_connected:.4f}")
    logger.info(f"  Density ratio:          {edge_density_connected/(edge_density_random+1e-10):.2f}x")
    
    logger.info("\nCorrelation structure:")
    logger.info(f"  Mean corr (1-hop):   {correlation_decay['one_hop']:.4f}")
    logger.info(f"  Mean corr (2-hop):   {correlation_decay['two_hop']:.4f}")
    logger.info(f"  Mean corr (no edge): {correlation_decay['no_edge']:.4f}")
    logger.info(f"  SNR (edge/non-edge): {snr:.3f}")
    
    # Store all metrics in dictionary
    enrichment_metrics = {
        'enrichment_I_N': enrichment_I_N,
        'enrichment_I_2': enrichment_I_2,
        'enrichment_I_HOC': enrichment_I_HOC,
        'enrichment_C3': enrichment_C3,
        'pvalue': pvalue,
        'cohens_d': cohens_d,
        'pct_above_median': pct_connected_above_median,
        'edge_density_random': edge_density_random,
        'edge_density_connected': edge_density_connected,
        'correlation_decay': correlation_decay,
        'snr': snr
    }
    
    return enrichment_metrics


def calculate_C3_for_random_samples(s, n_samples=100, N=10):
    """Calculate C_3 for random samples to enable comparison"""
    C_3_values = []
    rng = np.random.default_rng(seed=42)
    
    for _ in range(n_samples):
        idx = rng.choice(s.shape[1], size=N, replace=False)
        s_subset = s[:, idx]
        C_3 = compute_connected_triplets(s_subset)
        C_3_values.append(C_3)
    
    return np.array(C_3_values)


def calculate_edge_densities(s, edges):
    """Calculate average edge density for random and connected subgraphs"""
    n_neurons = s.shape[1]
    
    # Build adjacency matrix
    adj_matrix = np.zeros((n_neurons, n_neurons), dtype=bool)
    for j, i in edges.T:
        if i < n_neurons and j < n_neurons:
            adj_matrix[i, j] = True
            adj_matrix[j, i] = True
    
    # Random subgraphs
    rng = np.random.default_rng(seed=43)
    densities_random = []
    for _ in range(100):
        idx = rng.choice(n_neurons, size=10, replace=False)
        subgraph_adj = adj_matrix[np.ix_(idx, idx)]
        density = np.sum(subgraph_adj) / (10 * 9)  # Directed edges
        densities_random.append(density)
    
    # Connected subgraphs  
    subgraphs = sample_connected_subgraphs_fast(edges, n_neurons, n=10, n_samples=100)
    densities_connected = []
    for subset in subgraphs:
        subgraph_adj = adj_matrix[np.ix_(subset, subset)]
        density = np.sum(subgraph_adj) / (10 * 9)
        densities_connected.append(density)
    
    return np.mean(densities_random), np.mean(densities_connected)


def analyze_correlation_decay(s, edges, n_samples=1000):
    """Analyze how correlations decay with graph distance"""
    n_neurons = s.shape[1]
    
    # Build adjacency list
    from collections import defaultdict
    adj_list = defaultdict(set)
    for j, i in edges.T:
        if i < n_neurons and j < n_neurons:
            adj_list[i].add(j)
            adj_list[j].add(i)
    
    # Sample neuron pairs and compute correlations by distance
    rng = np.random.default_rng(seed=44)
    one_hop_corrs = []
    two_hop_corrs = []
    no_edge_corrs = []
    
    for _ in range(n_samples):
        i, j = rng.choice(n_neurons, size=2, replace=False)
        
        # Calculate correlation
        corr = np.corrcoef(s[:, i], s[:, j])[0, 1]
        
        # Determine graph distance
        if j in adj_list[i]:
            one_hop_corrs.append(abs(corr))
        else:
            # Check for 2-hop connection
            two_hop = False
            for neighbor in adj_list[i]:
                if j in adj_list[neighbor]:
                    two_hop = True
                    break
            
            if two_hop:
                two_hop_corrs.append(abs(corr))
            else:
                no_edge_corrs.append(abs(corr))
    
    return {
        'one_hop': np.mean(one_hop_corrs) if one_hop_corrs else 0,
        'two_hop': np.mean(two_hop_corrs) if two_hop_corrs else 0,
        'no_edge': np.mean(no_edge_corrs) if no_edge_corrs else 0
    }


def calculate_snr(s, edges):
    """Calculate signal-to-noise ratio (edge vs non-edge correlations)"""
    n_neurons = s.shape[1]
    
    # Build edge set for quick lookup
    edge_set = set()
    for j, i in edges.T:
        if i < n_neurons and j < n_neurons:
            edge_set.add((min(i,j), max(i,j)))
    
    # Sample correlations
    rng = np.random.default_rng(seed=45)
    edge_corrs = []
    non_edge_corrs = []
    
    for _ in range(1000):
        i, j = rng.choice(n_neurons, size=2, replace=False)
        i, j = min(i,j), max(i,j)
        
        corr = abs(np.corrcoef(s[:, i], s[:, j])[0, 1])
        
        if (i, j) in edge_set:
            edge_corrs.append(corr)
        else:
            non_edge_corrs.append(corr)
    
    mean_edge = np.mean(edge_corrs) if edge_corrs else 0
    mean_non_edge = np.mean(non_edge_corrs) if non_edge_corrs else 1e-10
    
    return mean_edge / mean_non_edge





def compute_entropy_analysis(s, delta_t, log_dir, logger, n_subsets=1000, N=10):
    """
    Compute information structure analysis on random neuron subsets.
    
    Args:
        s: Binary spike data [T, N_neurons] in {-1,+1}
        delta_t: Time step for rate conversion
        log_dir: Directory for saving results
        logger: Logger object
        n_subsets: Number of random subsets to analyze
        N: Size of each subset (neurons)
        seed: Random seed for reproducibility
        
    Returns:
        results_dict: Dictionary with analysis results
    """

    print('computing entropies...')
    
    rng = np.random.default_rng(seed=0)

    results = []
    for i in trange(n_subsets, desc="processing N-10 subsets"):
        idx = np.sort(rng.choice(s.shape[1], size=N, replace=False))
        s_subset = s[:, idx]

        result = analyze_N_10_information_structure(
            s_subset,
            logbase=2.0,
            alpha_joint=1e-3,
            alpha_marg=0.5,
            delta_t=delta_t,
            enforce_monotone=True
        )
        results.append(result)

    # Extract results
    INs = np.array([r.I_N for r in results])
    I2s = np.array([r.I2 for r in results])
    ratios = np.array([r.ratio for r in results])
    non_monotonic = np.array([r.count_non_monotonic for r in results])

    # Collect all patterns from all subsets
    all_observed = []
    all_predicted_pairwise = []
    all_predicted_independent = []

    for r in results:
        all_observed.extend(r.observed_rates)
        all_predicted_pairwise.extend(r.predicted_rates_pairwise)
        all_predicted_independent.extend(r.predicted_rates_independent)

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Panel A: Pattern rates scatter plot
    ax1.loglog(all_observed, all_predicted_pairwise, 'ro', alpha=0.1, markersize=0.5,
            label='pairwise model')
    ax1.loglog(all_observed, all_predicted_independent, 'go', alpha=0.1, markersize=0.5,
            label='independent model')
    ax1.plot([1e-4, 1e1], [1e-4, 1e1], 'w-', linewidth=2)
    ax1.set_xlim(1e-4, 1e1)
    ax1.set_ylim(1e-4, 1e1)
    ax1.set_xlabel('observed rate', fontsize=18)
    ax1.set_ylabel('predicted rate', fontsize=18)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Compute Jensen-Shannon divergences
    js_pairwise = []
    js_independent = []

    for r in results:
        obs = r.observed_rates + 1e-12
        pred_pair = r.predicted_rates_pairwise + 1e-12
        pred_indep = r.predicted_rates_independent + 1e-12

        def js_divergence(p, q):
            m = 0.5 * (p + q)
            return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

        js_pairwise.append(js_divergence(obs, pred_pair))
        js_independent.append(js_divergence(obs, pred_indep))

    # Panel B: JS divergence histogram
    ax2.hist(js_independent, bins=50, alpha=0.7, color='green', label='independent model', density=True)
    ax2.hist(js_pairwise, bins=50, alpha=0.7, color='red', label='pairwise model', density=True)
    ax2.set_xlabel('jensen-shannon divergence', fontsize=18)
    ax2.set_ylabel('probability density', fontsize=18)
    ax2.set_xscale('log')
    ax2.legend(fontsize=16)
    ax2.set_xlim(1e-2, 2e1)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    # Panel C: I_N vs ratio scatter
    ax3.scatter(INs, ratios, c='red', alpha=0.6, s=20, edgecolors='none', label='pairwise model')
    ax3.set_xlabel(r'multi-information $I_N$ (bits)', fontsize=18)
    ax3.set_ylabel(r'$I^{(2)}/I_N$', fontsize=18)
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.9, color='black', linestyle='--', alpha=0.5)
    ax3.tick_params(axis='both', which='major', labelsize=16)
    ax3.set_xlim(0, 150)
    ax3.set_ylim(0, 1.1)
    
    # Hide panel D
    ax4.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/Ising_rates.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Add I_HOC calculations
    I_HOCs = INs - I2s  # Higher-order contribution
    # Create results dictionary (add I_HOC_median for compatibility)
    results_dict = {
        'I_N': INs,
        'I2': I2s,
        'ratio': ratios,
        'count_non_monotonic': non_monotonic,
        'H_true': np.array([r.H_true for r in results]),
        'H_indep': np.array([r.H_indep for r in results]),
        'H_pair': np.array([r.H_pair for r in results]),
        'predicted_rates_pairwise': np.array([r.predicted_rates_pairwise for r in results]),
        'predicted_rates_independent': np.array([r.predicted_rates_independent for r in results]),
        'observed_rates': np.array([r.observed_rates for r in results]),
        'I_HOC_median': np.nanmedian(I_HOCs),
        'I_N_median': np.nanmedian(INs),
        'I_2_median': np.nanmedian(I2s),
        'ratio_median': np.nanmedian(ratios)
    }

    # Save results
    np.savez_compressed(f"{log_dir}/results/info_ratio_results.npz", **results_dict)

    # Log summary statistics
    q25_IN, q75_IN = np.nanpercentile(INs, [25, 75])
    q25_I2, q75_I2 = np.nanpercentile(I2s, [25, 75])
    q25_ratio, q75_ratio = np.nanpercentile(ratios, [25, 75])

    # Add I_HOC calculations
    I_HOCs = INs - I2s  # Higher-order contribution
    q25_IHOC, q75_IHOC = np.nanpercentile(I_HOCs, [25, 75])

    print(f"non monotonic ratio {non_monotonic.sum()} out of {n_subsets}")
    print(f"I_N:      median=\033[32m{np.nanmedian(INs):.3f}\033[0m,   IQR=[{q25_IN:.3f}, {q75_IN:.3f}],   std={np.nanstd(INs):.1f}")
    print(f"I2:       median=\033[32m{np.nanmedian(I2s):.3f}\033[0m,   IQR=[{q25_I2:.3f}, {q75_I2:.3f}],   std={np.nanstd(I2s):.1f}")
    print(f"I_HOC:    median=\033[32m{np.nanmedian(I_HOCs):.3f}\033[0m,   IQR=[{q25_IHOC:.3f}, {q75_IHOC:.3f}],   std={np.nanstd(I_HOCs):.3f}")
    print(f"I_2/I_N:  median=\033[32m{np.nanmedian(ratios):.3f}\033[0m,    IQR=[{q25_ratio:.3f}, {q75_ratio:.3f}],     std={np.nanstd(ratios):.3f}")

    logger.info(f"non monotonic ratio {non_monotonic.sum() / n_subsets:.2f}")
    logger.info(f"I_N:      median={np.nanmedian(INs):.3f},   IQR=[{q25_IN:.3f}, {q75_IN:.3f}],   std={np.nanstd(INs):.3f}")
    logger.info(f"I2:       median={np.nanmedian(I2s):.3f},   IQR=[{q25_I2:.3f}, {q75_I2:.3f}],   std={np.nanstd(I2s):.3f}")
    logger.info(f"I_HOC:    median={np.nanmedian(I_HOCs):.3f},   IQR=[{q25_IHOC:.3f}, {q75_IHOC:.3f}],   std={np.nanstd(I_HOCs):.3f}")
    logger.info(f"I_2/I_N:  median={np.nanmedian(ratios):.3f},   IQR=[{q25_ratio:.3f}, {q75_ratio:.3f}],   std={np.nanstd(ratios):.3f}")

    return results_dict

@dataclass
class InfoRatioResult:
    H_true: float
    H_indep: float
    H_pair: float
    I_N: float
    I2: float
    ratio: float
    count_non_monotonic: float
    observed_rates: np.ndarray
    predicted_rates_pairwise: np.ndarray  # P2 model (Ising)
    predicted_rates_independent: np.ndarray  # P1 model

def analyze_N_10_information_structure(S: np.ndarray,logbase: float = 2.0,alpha_joint: float = 1e-3,alpha_marg: float = 0.5,enforce_monotone: bool = False,ratio_eps: float = 1e-6,delta_t: float = 0.02,
) -> InfoRatioResult:
    """
    Compute information ratio and pattern rates for both pairwise and independent models.

    Parameters
    ----------
    S : np.ndarray, shape (T, N)
        Binary activity patterns in {-1, +1}

    Returns
    -------
    InfoRatioResult with pattern rates:
        - observed_rates: empirical probability of each 2^N pattern
        - predicted_rates_pairwise: Ising model (P2) probability of each pattern
        - predicted_rates_independent: Independent model (P1) probability of each pattern
    """
    H_true = entropy_true_pseudocount(S, logbase, alpha_joint)
    H_ind = entropy_indep_bernoulli_jeffreys(S, logbase, alpha_marg)

    h, J, H_pair = entropy_exact_ising(S)

    if (H_pair > H_ind) | (H_true > H_pair):
        count_non_monotonic = 1
    else:
        count_non_monotonic = 0

    if enforce_monotone:
        H_pair = min(H_pair, H_ind)
        H_true = min(H_true, H_pair)

    I_N = H_ind - H_true
    I2 = H_ind - H_pair
    ratio = I2 / I_N if I_N > ratio_eps else float('nan')

    # Pattern rate computation (always done)
    T, N = S.shape

    # Convert {-1,+1} to {0,1} for pattern indexing
    S_binary = (S + 1) // 2  # shape (T, N)

    # Compute pattern indices (each pattern -> integer 0 to 2^N-1)
    powers = 2 ** np.arange(N)  # [1, 2, 4, 8, ...]
    pattern_indices = S_binary @ powers  # shape (T,)

    # Count empirical pattern occurrences
    pattern_counts = np.bincount(pattern_indices, minlength=2 ** N)
    observed_rates = pattern_counts.astype(float) / T

    # Compute individual neuron firing probabilities for independent model
    p_i = np.mean(S_binary, axis=0)  # shape (N,)

    # Compute model predictions for all 2^N patterns
    predicted_rates_pairwise = np.zeros(2 ** N)
    predicted_rates_independent = np.zeros(2 ** N)

    # Generate all possible binary patterns
    for pattern_idx in range(2 ** N):
        # Convert pattern index back to binary array
        binary_pattern = np.array([(pattern_idx >> i) & 1 for i in range(N)])

        # PAIRWISE MODEL (P2): Ising model
        sigma = 2 * binary_pattern - 1  # Convert to {-1, +1}
        field_energy = -np.sum(h * sigma)
        coupling_energy = -0.5 * np.sum(J * np.outer(sigma, sigma))
        energy = field_energy + coupling_energy
        predicted_rates_pairwise[pattern_idx] = np.exp(-energy)

        # INDEPENDENT MODEL (P1): Product of individual probabilities
        prob_independent = 1.0
        for i in range(N):
            if binary_pattern[i] == 1:
                prob_independent *= p_i[i]
            else:
                prob_independent *= (1.0 - p_i[i])
        predicted_rates_independent[pattern_idx] = prob_independent

    # Normalize pairwise model to get probabilities
    Z_pairwise = np.sum(predicted_rates_pairwise)
    predicted_rates_pairwise = predicted_rates_pairwise / Z_pairwise

    # Independent model already normalized (probabilities sum to 1)

    return InfoRatioResult(
        H_true=H_true,
        H_indep=H_ind,
        H_pair=H_pair,
        I_N=I_N / delta_t,
        I2=I2 / delta_t,
        ratio=ratio,
        count_non_monotonic=count_non_monotonic,
        observed_rates=observed_rates / delta_t,  # NOW in s^-1
        predicted_rates_pairwise=predicted_rates_pairwise / delta_t,  # NOW in s^-1
        predicted_rates_independent=predicted_rates_independent / delta_t
    )

def plugin_entropy(p: np.ndarray, logbase: float = math.e) -> float:
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    return H / math.log(logbase)

def miller_madow_correction(k: int, n: int, logbase: float = math.e) -> float:
    if n <= 0: return 0.0
    return ((k - 1) / (2.0 * n)) / math.log(logbase)

def empirical_entropy_true(S: np.ndarray, logbase: float = math.e) -> float:
    T, N = S.shape
    X = ((S + 1) // 2).astype(np.int8)
    keys = X @ (1 << np.arange(N, dtype=np.int64))
    counts = np.bincount(keys, minlength=1 << N).astype(np.float64)
    p = counts[counts > 0] / T
    H = plugin_entropy(p, logbase)
    H += miller_madow_correction(len(p), T, logbase)
    return H

def entropy_true_pseudocount(S: np.ndarray, logbase: float, alpha: float) -> float:
    T, N = S.shape
    X = ((S + 1) // 2).astype(np.int8)
    keys = X @ (1 << np.arange(N, dtype=np.int64))
    counts = np.bincount(keys, minlength=1 << N).astype(np.float64) + alpha
    p = counts / counts.sum()
    H = plugin_entropy(p, logbase)
    H += miller_madow_correction(np.count_nonzero(p), int(counts.sum()), logbase)
    return H

def entropy_indep_bernoulli_jeffreys(S: np.ndarray, logbase: float, alpha: float) -> float:
    T, N = S.shape
    X = ((S + 1) // 2).astype(np.int8)
    H = 0.0
    for i in range(N):
        k1 = X[:, i].sum()
        p1 = (k1 + alpha) / (T + 2 * alpha)
        p0 = 1.0 - p1
        h = 0.0
        if p1 > 0: h -= p1 * math.log(p1)
        if p0 > 0: h -= p0 * math.log(p0)
        H += h / math.log(logbase)
        H += miller_madow_correction(2, T, logbase)
    return H

def entropy_exact_ising(S, max_iter=200, lr=1.0, lam=0.0, tol=1e-6, logbase=2.0, verbose=False):
    """
    Maximum likelihood fit of pairwise Ising with exact enumeration.
    S in {-1,+1}^{T x N}. Returns h, J, H_pair.
    """
    S = S.astype(np.int8)
    # Ensure {-1,+1}
    uniq = np.unique(S)
    if set(uniq.tolist()) == {0,1}:
        S = 2*S - 1
    elif not set(uniq.tolist()) <= {-1,1}:
        raise ValueError("S must be binary in {-1,+1} or {0,1}.")

    T, N = S.shape
    m_data = S.mean(axis=0)                                 # ⟨s_i⟩_data
    C_data = (S[:, :, None] * S[:, None, :]).mean(axis=0)   # ⟨s_i s_j⟩_data
    np.fill_diagonal(C_data, 1.0)

    # Initialize (PL result or zeros). Start from PL helps, but zeros also work for N=10.
    h = np.zeros(N, dtype=np.float64)
    J = np.zeros((N, N), dtype=np.float64)

    X = enumerate_states_pm1(N)

    def neg_loglike_and_grad(h, J):
        # ℓ(θ) = ∑_t [ h·s + 1/2 s^T J s ] - T * log Z(θ)
        # We return -ℓ for minimization.
        p, _, _ = model_probs(h, J, X)
        m_mod = p @ X
        C_mod = np.tensordot(p, X[:, :, None] * X[:, None, :], axes=(0,0))
        np.fill_diagonal(C_mod, 1.0)
        # Gradients of -ℓ (so we can do standard gradient DESCENT on f = -ℓ)
        # For ascent on ℓ, use the negative of these.
        g_h = -(m_data - m_mod) + lam * h
        g_J = -(C_data - C_mod) + lam * J
        np.fill_diagonal(g_J, 0.0)
        # Negative log-likelihood:
        # -ℓ = -T*( h·⟨s⟩_data + 1/2 tr(J C_data) ) + T*logZ + reg
        # The constant “−T * H_emp” is irrelevant, so omitted.
        # We don’t compute f precisely here; we only need gradients for optimization & monitoring.
        return g_h, g_J

    # Backtracking line search on ℓ (maximize ℓ)
    prev_ll = -np.inf
    for it in range(1, max_iter+1):
        # Compute model stats and log-likelihood for monitoring
        p, _, logZ = model_probs(h, J, X)
        p @ X
        C_mod = np.tensordot(p, X[:, :, None] * X[:, None, :], axes=(0,0))
        np.fill_diagonal(C_mod, 1.0)

        ll = T * (m_data @ h + 0.5 * np.sum(J * C_data) - logZ) - 0.5*lam*(h@h + (J*J).sum())
        # Gradient for ascent on ℓ:
        g_h, g_J = neg_loglike_and_grad(h, J)
        g_h *= -1.0
        g_J *= -1.0

        step = lr
        # Backtracking to ensure ll increases
        for _ in range(30):
            h_new = h + step * g_h
            J_new = J + step * g_J
            J_new = 0.5 * (J_new + J_new.T)
            np.fill_diagonal(J_new, 0.0)
            p_new, _, logZ_new = model_probs(h_new, J_new, X)
            ll_new = T * (m_data @ h_new + 0.5 * np.sum(J_new * C_data) - logZ_new) - 0.5*lam*(h_new@h_new + (J_new*J_new).sum())
            if ll_new >= ll:
                break
            step *= 0.5

        h, J, ll = h_new, J_new, ll_new

        # Convergence checks
        grad_norm = np.sqrt((g_h @ g_h) + (g_J*g_J).sum())
        if verbose and (it % 10 == 0 or it == 1):
            print(f"[it {it:3d}] ll/T = {ll/T:.6f}  step={step:.3e}  ||grad||={grad_norm:.3e}")

        if abs(ll - prev_ll) / (1 + abs(prev_ll)) < tol and grad_norm < 1e-5:
            break
        prev_ll = ll

    H_pair = entropy_from_model(h, J, logbase=logbase)
    return h, J, H_pair

def enumerate_states_pm1(N):
    # All 2^N states in {-1,+1}^N
    X = np.array(list(product([-1, 1], repeat=N)), dtype=np.int8)
    return X  # shape [2^N, N]

def energy(h, J, X):
    # E(s) = - h·s - 1/2 s^T J s  (assumes J symmetric, diag(J)=0)
    # X: [M, N], h: [N], J: [N, N]
    # Ensure diagonal is zero to avoid double-counting
    J_offdiag = J.copy()
    np.fill_diagonal(J_offdiag, 0)
    lin = X @ h
    quad = np.einsum('bi,ij,bj->b', X, J_offdiag, X)
    return -(lin + 0.5 * quad)

def log_partition(E):
    # log Z = logsumexp(-E) if E is defined as +energy; here E is already energy
    # We defined p ∝ exp(-E), so we need logsumexp(-E)
    m = (-E).max()
    return m + np.log(np.exp((-E - m)).sum())

def model_probs(h, J, X):
    E = energy(h, J, X)
    logZ = log_partition(E)
    logp = -E - logZ
    p = np.exp(logp)
    return p, E, logZ

def model_moments_exact(h, J):
    N = len(h)
    X = enumerate_states_pm1(N)
    p, _, _ = model_probs(h, J, X)
    m = p @ X                          # ⟨s_i⟩
    C = (X[:, :, None] * X[:, None, :])  # [M, N, N]
    C = np.tensordot(p, C, axes=(0,0))   # ⟨s_i s_j⟩
    return m, C

def entropy_from_model(h, J, logbase=2.0):
    X = enumerate_states_pm1(len(h))
    p, E, logZ = model_probs(h, J, X)
    # H = -sum p log p
    H_nats = -(p * (np.log(p + 1e-300))).sum()
    return H_nats / log(logbase)








def sample_connected_subgraphs_fast(edges, n_neurons=13741, n=10, n_samples=100):
   """Faster implementation using adjacency list"""
   
   # Build adjacency list
   adj_list = defaultdict(set)
   for j, i in edges.T:
       adj_list[i].add(j)
       adj_list[j].add(i)  # Treat as undirected for connectivity
   
   subgraphs = []
   attempts = 0
   max_attempts = n_samples * 10
   pbar = tqdm(total=n_samples, desc="sampling connected subgraphs")
   while len(subgraphs) < n_samples and attempts < max_attempts:
       attempts += 1
       # Start from random neuron with connections
       connected_neurons = list(adj_list.keys())
       if not connected_neurons:
           break
       seed = np.random.choice(connected_neurons)
       # BFS to find 2-hop neighborhood
       visited = {seed}
       current_level = {seed}
       for hop in range(2):
           next_level = set()
           for node in current_level:
               next_level.update(adj_list[node])
           visited.update(next_level)
           current_level = next_level
       if len(visited) >= n:
           subset = np.random.choice(list(visited), n, replace=False)
           subgraphs.append(subset)
           pbar.update(1)
   pbar.close()
   
   return np.array(subgraphs)

def analyze_higher_order_correlations(s, edges, delta_t, log_dir, logger, n_subsets=100, N=10):
    """Analyze HOC using connectivity-aware sampling"""
    
    # Sample connected subgraphs
    subgraphs = sample_connected_subgraphs_fast(edges, n_neurons=s.shape[1], n=N, n_samples=n_subsets)
    
    # Storage for metrics
    I_Ns = []
    I_2s = []
    I_HOCs = []
    C_3s = []  # Connected triple correlations
    ratios = []
    
    for subset in subgraphs:
        s_subset = s[:, subset]
        
        # Use the existing function!
        result = analyze_N_10_information_structure(
            s_subset,
            logbase=2.0,
            alpha_joint=1e-3,
            alpha_marg=0.5,
            delta_t=delta_t,
            enforce_monotone=True
        )
        
        I_N = result.I_N
        I_2 = result.I2
        I_HOC = I_N - I_2
        
        I_Ns.append(I_N)
        I_2s.append(I_2)
        I_HOCs.append(I_HOC)
        ratios.append(result.ratio)
        
        # Connected triple correlations (genuine 3-body)
        C_3 = compute_connected_triplets(s_subset)
        C_3s.append(C_3)
    
    # Convert to arrays
    I_Ns = np.array(I_Ns)
    I_2s = np.array(I_2s)
    I_HOCs = np.array(I_HOCs)
    C_3s = np.array(C_3s)
    ratios = np.array(ratios)
    
    # Compute percentiles
    q25_IN, q75_IN = np.nanpercentile(I_Ns, [25, 75])
    q25_I2, q75_I2 = np.nanpercentile(I_2s, [25, 75])
    q25_IHOC, q75_IHOC = np.nanpercentile(I_HOCs, [25, 75])
    q25_ratio, q75_ratio = np.nanpercentile(ratios, [25, 75])
    
    # Print to console (with colors)
    print(f"connected subgraph analysis (N={N}):")
    print(f"I_N:      median=\033[32m{np.nanmedian(I_Ns):.3f}\033[0m,   IQR=[{q25_IN:.3f}, {q75_IN:.3f}],   std={np.nanstd(I_Ns):.1f}")
    print(f"I2:       median=\033[32m{np.nanmedian(I_2s):.3f}\033[0m,   IQR=[{q25_I2:.3f}, {q75_I2:.3f}],   std={np.nanstd(I_2s):.1f}")
    print(f"I_HOC:    median=\033[32m{np.nanmedian(I_HOCs):.3f}\033[0m,   IQR=[{q25_IHOC:.3f}, {q75_IHOC:.3f}],   std={np.nanstd(I_HOCs):.3f}")
    print(f"I_2/I_N:  median=\033[32m{np.nanmedian(ratios):.3f}\033[0m,    IQR=[{q25_ratio:.3f}, {q75_ratio:.3f}],     std={np.nanstd(ratios):.3f}")
    print(f"C_3:      mean=\033[32m{np.mean(C_3s):.4f}\033[0m ± {np.std(C_3s):.4f}")
    
    # Log to file (consistent format)
    logger.info(f"Connected subgraph analysis (N={N}):")
    logger.info(f"I_N:      median={np.nanmedian(I_Ns):.3f},   IQR=[{q25_IN:.3f}, {q75_IN:.3f}],   std={np.nanstd(I_Ns):.3f}")
    logger.info(f"I2:       median={np.nanmedian(I_2s):.3f},   IQR=[{q25_I2:.3f}, {q75_I2:.3f}],   std={np.nanstd(I_2s):.3f}")
    logger.info(f"I_HOC:    median={np.nanmedian(I_HOCs):.3f},   IQR=[{q25_IHOC:.3f}, {q75_IHOC:.3f}],   std={np.nanstd(I_HOCs):.3f}")
    logger.info(f"I_2/I_N:  median={np.nanmedian(ratios):.3f},   IQR=[{q25_ratio:.3f}, {q75_ratio:.3f}],   std={np.nanstd(ratios):.3f}")
    logger.info(f"C_3:      mean={np.mean(C_3s):.4f} ± {np.std(C_3s):.4f}")
    
    # Compute statistics for return
    results = {
        'I_N': I_Ns,
        'I2': I_2s,
        'ratio': ratios,
        'I_N_median': np.nanmedian(I_Ns),
        'I_2_median': np.nanmedian(I_2s),
        'I_HOC_median': np.nanmedian(I_HOCs),
        'ratio_median': np.nanmedian(ratios),
        'C_3_mean': np.mean(C_3s),
        'C_3_std': np.std(C_3s)
    }
    
    return results
    """Analyze HOC using connectivity-aware sampling"""
    
    # Sample connected subgraphs
    subgraphs = sample_connected_subgraphs_fast(edges, n_neurons=s.shape[1], n=N, n_samples=n_subsets)
    
    # Storage for metrics
    I_Ns = []
    I_2s = []
    I_HOCs = []
    C_3s = []  # Connected triple correlations
    ratios = []
    
    for idx in trange(len(subgraphs), desc="processing connected subgraphs"):
        subset = subgraphs[idx]
        s_subset = s[:, subset]
        
        # Use the existing function!
        result = analyze_N_10_information_structure(
            s_subset,
            logbase=2.0,
            alpha_joint=1e-3,
            alpha_marg=0.5,
            delta_t=delta_t,
            enforce_monotone=True
        )
        
        I_N = result.I_N
        I_2 = result.I2
        I_HOC = I_N - I_2
        
        I_Ns.append(I_N)
        I_2s.append(I_2)
        I_HOCs.append(I_HOC)
        ratios.append(result.ratio)
        
        # Connected triple correlations (genuine 3-body)
        C_3 = compute_connected_triplets(s_subset)
        C_3s.append(C_3)
    
    # Convert to arrays
    I_Ns = np.array(I_Ns)
    I_2s = np.array(I_2s)
    I_HOCs = np.array(I_HOCs)
    C_3s = np.array(C_3s)
    ratios = np.array(ratios)
    
    # Compute statistics - match the structure from compute_entropy_analysis
    results = {
        'I_N': I_Ns,
        'I2': I_2s,
        'ratio': ratios,
        'I_N_median': np.nanmedian(I_Ns),
        'I_2_median': np.nanmedian(I_2s),
        'I_HOC_median': np.nanmedian(I_HOCs),
        'ratio_median': np.nanmedian(ratios),
        'C_3_mean': np.mean(C_3s),
        'C_3_std': np.std(C_3s)
    }
    
    # Log results
    print(f"I_N:    median=\033[32m{results['I_N_median']:.4f}\033[0m")
    print(f"I_2:    median=\033[32m{results['I_2_median']:.4f}\033[0m")
    print(f"I_HOC:  median=\033[32m{results['I_HOC_median']:.4f}\033[0m")
    print(f"I_2/I_N:   {results['ratio_median']:.3f}")
    print(f"C_3:   {results['C_3_mean']:.4f} ± {results['C_3_std']:.4f}")
    logger.info(f"I_N:   {results['I_N_median']:.4f} bits")
    logger.info(f"I_2:   {results['I_2_median']:.4f} bits")
    logger.info(f"I_HOC: {results['I_HOC_median']:.4f} bits")
    logger.info(f"I_2/I_N:   {results['ratio_median']:.3f}")
    logger.info(f"C_3:   {results['C_3_mean']:.4f} ± {results['C_3_std']:.4f}")
    
    return results

def compute_connected_triplets(s_subset):
   """Compute connected correlation for all triplets"""
   N = s_subset.shape[1]
   if N < 3:
       return 0.0
   
   # Convert to {0,1}
   s_binary = (s_subset + 1) // 2
   
   C_3_values = []
   
   # Sample random triplets (exhaustive for small N)
   n_triplets = min(100, N*(N-1)*(N-2)//6)
   
   for _ in range(n_triplets):
       i, j, k = np.random.choice(N, 3, replace=False)
       
       # Means
       μ_i = s_binary[:, i].mean()
       μ_j = s_binary[:, j].mean()
       μ_k = s_binary[:, k].mean()
       
       # Pairwise covariances
       C_ij = np.cov(s_binary[:, i], s_binary[:, j])[0, 1]
       C_ik = np.cov(s_binary[:, i], s_binary[:, k])[0, 1]
       C_jk = np.cov(s_binary[:, j], s_binary[:, k])[0, 1]
       
       # Triple correlation
       μ_ijk = (s_binary[:, i] * s_binary[:, j] * s_binary[:, k]).mean()
       
       # Connected correlation (cumulant)
       C_ijk = μ_ijk - μ_i*μ_j*μ_k - μ_i*C_jk - μ_j*C_ik - μ_k*C_ij
       
       C_3_values.append(abs(C_ijk))
   
   return np.mean(C_3_values)







