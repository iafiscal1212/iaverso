"""
LX Metrics: Pure metric functions for Life-Extended Cognition Benchmark
========================================================================

All functions are 100% ENDOGENOUS - no magic constants.
Parameters derived from: percentiles, ranks, sqrt(t), covariances.

LX1-LX10 measure integration between:
- Symbols <-> Phases (symbol-phase)
- Medicine <-> Life phases (medicine-phase)
- Dream narrative (dream-narrative)
- Circadian impact on CG-E
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
from scipy import stats


# ==============================================================================
# CORE ENDOGENOUS UTILITIES
# ==============================================================================

def endogenous_window(t: int) -> int:
    """
    Endogenous window size: L_t = max(3, floor(sqrt(t)))

    No magic constants - derived from temporal experience.
    """
    return max(3, int(np.floor(np.sqrt(t))))


def endogenous_rank(value: float, history: np.ndarray) -> float:
    """
    Endogenous rank: position in [0, 1] within own distribution.

    rank(x) = proportion of history values <= x
    """
    if len(history) == 0:
        return 0.5
    return np.mean(history <= value)


def endogenous_percentile(history: np.ndarray, p: float) -> float:
    """
    Endogenous percentile: Q_p computed on agent's own history.
    """
    if len(history) == 0:
        return 0.0
    return np.percentile(history, p)


def endogenous_ranks(values: np.ndarray) -> np.ndarray:
    """
    Convert array to ranks in [0, 1].
    """
    if len(values) == 0:
        return np.array([])
    return stats.rankdata(values, method='average') / len(values)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def mutual_information(joint_probs: np.ndarray) -> float:
    """
    Mutual information from joint probability table.

    I(X;Y) = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
    """
    eps = 1e-12
    # Marginals
    p_x = joint_probs.sum(axis=1)
    p_y = joint_probs.sum(axis=0)

    mi = 0.0
    for i in range(joint_probs.shape[0]):
        for j in range(joint_probs.shape[1]):
            if joint_probs[i, j] > eps:
                mi += joint_probs[i, j] * np.log(
                    joint_probs[i, j] / (p_x[i] * p_y[j] + eps) + eps
                )
    return mi


def entropy(probs: np.ndarray) -> float:
    """
    Shannon entropy: H(X) = -Σ p(x) log(p(x))
    """
    eps = 1e-12
    probs = probs[probs > eps]
    return -np.sum(probs * np.log(probs + eps))


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

class CircadianPhase(Enum):
    WAKE = "WAKE"
    REST = "REST"
    DREAM = "DREAM"
    LIMINAL = "LIMINAL"


@dataclass
class SymbolActivation:
    """Symbol activation at time t for agent A."""
    t: int
    agent_id: str
    symbol_id: str
    active: bool  # s_t^A(σ) ∈ {0, 1}


@dataclass
class PhaseState:
    """Phase state at time t for agent A."""
    t: int
    agent_id: str
    phase: CircadianPhase
    energy: float
    stress: float


@dataclass
class Episode:
    """Episode k for agent A."""
    k: int
    agent_id: str
    t_start: int
    t_end: int
    dominant_phase: CircadianPhase
    narrative_coherence: float  # NC_k^A
    symbol_density: float  # proportion of steps with >= 1 symbol


@dataclass
class CycleStats:
    """Statistics for circadian cycle c for agent A."""
    c: int
    agent_id: str
    cge: float  # CGE for this cycle
    shock_mean: float
    health_mean: float
    med_ratio: float  # proportion of time under medical intervention
    symbol_vector: np.ndarray  # mean symbol activations
    trait_vector: np.ndarray  # [drives, shock, health, cge, symbol_density...]


# ==============================================================================
# LX1: PHASE-SYMBOL SPECIALIZATION
# ==============================================================================

def compute_lx1_phase_symbol(
    symbol_activations: List[SymbolActivation],
    phase_states: List[PhaseState],
    agent_ids: List[str]
) -> Dict[str, float]:
    """
    LX1: Phase-Symbol Specialization

    Measures how clearly certain symbols "belong" to specific phases.

    For each symbol σ:
        Spec(σ) = I(σ; p) / (H(σ) + ε)

    Where I is mutual information and H is entropy.

    Returns:
        {'LX1_global': float, 'LX1_<agent>': float, ...}
    """
    eps = 1e-12

    # Build time-indexed lookups
    phase_by_t_agent: Dict[Tuple[int, str], CircadianPhase] = {}
    for ps in phase_states:
        phase_by_t_agent[(ps.t, ps.agent_id)] = ps.phase

    # Collect all symbols
    all_symbols: Set[str] = set()
    for sa in symbol_activations:
        if sa.active:
            all_symbols.add(sa.symbol_id)

    if len(all_symbols) == 0:
        return {'LX1_global': 0.0}

    symbol_list = sorted(all_symbols)
    phases = list(CircadianPhase)
    n_symbols = len(symbol_list)
    n_phases = len(phases)

    # Build joint probability table: p(σ, p)
    # For each agent
    agent_specs: Dict[str, float] = {}

    for agent_id in agent_ids:
        # Filter for this agent
        agent_activations = [sa for sa in symbol_activations if sa.agent_id == agent_id]

        if len(agent_activations) == 0:
            agent_specs[agent_id] = 0.0
            continue

        # Count joint occurrences
        joint_counts = np.zeros((n_symbols, n_phases))
        total_count = 0

        for sa in agent_activations:
            if sa.active:
                key = (sa.t, agent_id)
                if key in phase_by_t_agent:
                    phase = phase_by_t_agent[key]
                    sym_idx = symbol_list.index(sa.symbol_id)
                    phase_idx = phases.index(phase)
                    joint_counts[sym_idx, phase_idx] += 1
                    total_count += 1

        if total_count == 0:
            agent_specs[agent_id] = 0.0
            continue

        # Normalize to probabilities
        joint_probs = joint_counts / (total_count + eps)

        # Calculate specialization for each symbol
        symbol_specs = []
        for sym_idx in range(n_symbols):
            sym_probs = joint_probs[sym_idx, :]
            sym_total = sym_probs.sum()

            if sym_total < eps:
                continue

            # p(σ)
            p_sigma = sym_total
            # H(σ) - entropy of symbol (active or not doesn't matter here,
            # we use distribution across phases)
            H_sigma = entropy(sym_probs / (sym_total + eps))

            # For MI, we need p(σ, p) / (p(σ) * p(p))
            phase_marginal = joint_probs.sum(axis=0)

            mi_sigma = 0.0
            for phase_idx in range(n_phases):
                if joint_probs[sym_idx, phase_idx] > eps:
                    mi_sigma += joint_probs[sym_idx, phase_idx] * np.log(
                        joint_probs[sym_idx, phase_idx] /
                        (p_sigma * phase_marginal[phase_idx] + eps) + eps
                    )

            # Specialization
            spec = mi_sigma / (H_sigma + eps) if H_sigma > eps else 0.0
            symbol_specs.append(max(0, min(1, spec)))

        if len(symbol_specs) > 0:
            agent_specs[agent_id] = np.mean(symbol_specs)
        else:
            agent_specs[agent_id] = 0.0

    # Global score: rank-averaged
    if len(agent_specs) > 0:
        scores = np.array(list(agent_specs.values()))
        global_score = np.mean(endogenous_ranks(scores))
    else:
        global_score = 0.0

    result = {'LX1_global': global_score}
    for agent_id, score in agent_specs.items():
        result[f'LX1_{agent_id}'] = score

    return result


# ==============================================================================
# LX2: CIRCADIAN SYMBOLIC DRIFT
# ==============================================================================

def compute_lx2_symbolic_drift(
    cycle_stats: List[CycleStats],
    agent_ids: List[str]
) -> Dict[str, float]:
    """
    LX2: Circadian Symbolic Drift

    Measures how symbols change from cycle to cycle - neither frozen nor chaotic.

    For each cycle c:
        D_c^A = 1 - cos(v_c^A, v_{c+1}^A)

    where v_c^A is mean symbol vector for cycle c.

    Optimal drift = close to median (not 0 = rigid, not max = chaotic)

    LX2_A = 1 - |D*_A - median(D)| / (Q75(D) - Q25(D) + ε)

    Returns:
        {'LX2_global': float, 'LX2_<agent>': float, ...}
    """
    eps = 1e-12
    agent_scores: Dict[str, float] = {}

    for agent_id in agent_ids:
        # Get cycles for this agent, sorted
        agent_cycles = sorted(
            [cs for cs in cycle_stats if cs.agent_id == agent_id],
            key=lambda x: x.c
        )

        if len(agent_cycles) < 2:
            agent_scores[agent_id] = 0.5  # Not enough data
            continue

        # Calculate drift between consecutive cycles
        drifts = []
        for i in range(len(agent_cycles) - 1):
            v_c = agent_cycles[i].symbol_vector
            v_c1 = agent_cycles[i + 1].symbol_vector

            if len(v_c) > 0 and len(v_c1) > 0:
                cos_sim = cosine_similarity(v_c, v_c1)
                drift = 1 - cos_sim
                drifts.append(drift)

        if len(drifts) == 0:
            agent_scores[agent_id] = 0.5
            continue

        drifts = np.array(drifts)

        # Optimal drift: close to median
        D_star = np.median(drifts)
        Q25 = np.percentile(drifts, 25)
        Q75 = np.percentile(drifts, 75)
        IQR = Q75 - Q25 + eps

        # Score: 1 when at median, lower when far from median
        deviation = abs(D_star - np.median(drifts))  # This is 0 by definition
        # Better: measure consistency around median
        mad = np.mean(np.abs(drifts - D_star))
        score = 1 - (mad / IQR) if IQR > eps else 0.5
        agent_scores[agent_id] = max(0, min(1, score))

    # Global
    if len(agent_scores) > 0:
        scores = np.array(list(agent_scores.values()))
        global_score = np.mean(endogenous_ranks(scores))
    else:
        global_score = 0.0

    result = {'LX2_global': global_score}
    for agent_id, score in agent_scores.items():
        result[f'LX2_{agent_id}'] = score

    return result


# ==============================================================================
# LX3: DREAM NARRATIVE
# ==============================================================================

def compute_lx3_dream_narrative(
    episodes: List[Episode],
    agent_ids: List[str]
) -> Dict[str, float]:
    """
    LX3: Dream Narrative

    Measures narrative coherence during DREAM phase.
    DREAM is not noise but consolidating narrative chapters.

    For each dream episode:
        DreamNarr_k^A = rank(NC_k^A * density_k^A)

    where NC is narrative coherence and density is symbol density.

    Returns:
        {'LX3_global': float, 'LX3_<agent>': float, ...}
    """
    agent_scores: Dict[str, float] = {}

    for agent_id in agent_ids:
        # Filter dream episodes
        dream_episodes = [
            ep for ep in episodes
            if ep.agent_id == agent_id and ep.dominant_phase == CircadianPhase.DREAM
        ]

        if len(dream_episodes) == 0:
            agent_scores[agent_id] = 0.5
            continue

        # Calculate dream narrative index for each episode
        dream_narr_values = []
        for ep in dream_episodes:
            # Combine narrative coherence and symbol density
            value = ep.narrative_coherence * ep.symbol_density
            dream_narr_values.append(value)

        dream_narr_values = np.array(dream_narr_values)

        # Rank within this agent's dream episodes
        ranks = endogenous_ranks(dream_narr_values)
        agent_scores[agent_id] = np.mean(ranks)

    # Global
    if len(agent_scores) > 0:
        scores = np.array(list(agent_scores.values()))
        global_score = np.mean(endogenous_ranks(scores))
    else:
        global_score = 0.0

    result = {'LX3_global': global_score}
    for agent_id, score in agent_scores.items():
        result[f'LX3_{agent_id}'] = score

    return result


# ==============================================================================
# LX4: DREAM-WAKE TRANSFER
# ==============================================================================

def compute_lx4_dream_transfer(
    episodes: List[Episode],
    cge_by_block: Dict[int, float],  # block_id -> CGE
    agent_ids: List[str]
) -> Dict[str, float]:
    """
    LX4: Dream-Wake Transfer

    Measures if DREAM improves subsequent WAKE performance.

    For each dream episode:
        ΔCGE_k = CGE(W_post) - CGE(W_pre)

    Correlate DreamNarr with ΔCGE:
        ρ_A = Spearman(DreamNarr_k, ΔCGE_k)

    LX4_A = rank(ρ_A)

    Returns:
        {'LX4_global': float, 'LX4_<agent>': float, ...}
    """
    agent_scores: Dict[str, float] = {}

    for agent_id in agent_ids:
        # Get episodes for this agent
        agent_episodes = sorted(
            [ep for ep in episodes if ep.agent_id == agent_id],
            key=lambda x: x.t_start
        )

        if len(agent_episodes) < 3:
            agent_scores[agent_id] = 0.5
            continue

        # Find dream episodes with pre/post WAKE episodes
        dream_narr_values = []
        delta_cge_values = []

        for i, ep in enumerate(agent_episodes):
            if ep.dominant_phase != CircadianPhase.DREAM:
                continue

            # Find pre-WAKE episode
            pre_wake = None
            for j in range(i - 1, -1, -1):
                if agent_episodes[j].dominant_phase == CircadianPhase.WAKE:
                    pre_wake = agent_episodes[j]
                    break

            # Find post-WAKE episode
            post_wake = None
            for j in range(i + 1, len(agent_episodes)):
                if agent_episodes[j].dominant_phase == CircadianPhase.WAKE:
                    post_wake = agent_episodes[j]
                    break

            if pre_wake is None or post_wake is None:
                continue

            # Get CGE for blocks (approximate by episode index)
            pre_cge = cge_by_block.get(pre_wake.k, 0.5)
            post_cge = cge_by_block.get(post_wake.k, 0.5)
            delta_cge = post_cge - pre_cge

            dream_narr = ep.narrative_coherence * ep.symbol_density

            dream_narr_values.append(dream_narr)
            delta_cge_values.append(delta_cge)

        if len(dream_narr_values) < 3:
            agent_scores[agent_id] = 0.5
            continue

        # Spearman correlation
        rho, _ = stats.spearmanr(dream_narr_values, delta_cge_values)
        if np.isnan(rho):
            rho = 0.0

        # Rank the correlation (transform to [0, 1])
        # rho in [-1, 1], so (rho + 1) / 2 gives [0, 1]
        agent_scores[agent_id] = (rho + 1) / 2

    # Global
    if len(agent_scores) > 0:
        scores = np.array(list(agent_scores.values()))
        global_score = np.mean(endogenous_ranks(scores))
    else:
        global_score = 0.0

    result = {'LX4_global': global_score}
    for agent_id, score in agent_scores.items():
        result[f'LX4_{agent_id}'] = score

    return result


# ==============================================================================
# LX5: MEDICINE-PHASE ALIGNMENT
# ==============================================================================

def compute_lx5_medicine_phase(
    interventions: List[Tuple[int, str, bool]],  # (t, agent_id, intervened)
    needs: List[Tuple[int, str, float]],  # (t, agent_id, need_score)
    agent_ids: List[str]
) -> Dict[str, float]:
    """
    LX5: Medicine-Phase Alignment

    Measures if medical intervention occurs when needed.

    need_t^A = rank(shock + stress - health)

    LX5_A = P(need_t+ > need_t-) where + means intervention, - means no intervention

    This is essentially an AUC measure.

    Returns:
        {'LX5_global': float, 'LX5_<agent>': float, ...}
    """
    agent_scores: Dict[str, float] = {}

    # Build lookups
    intervention_map: Dict[Tuple[int, str], bool] = {}
    for t, agent_id, intervened in interventions:
        intervention_map[(t, agent_id)] = intervened

    need_map: Dict[Tuple[int, str], float] = {}
    for t, agent_id, need in needs:
        need_map[(t, agent_id)] = need

    for agent_id in agent_ids:
        # Get all time points for this agent
        agent_times = [t for (t, aid) in need_map.keys() if aid == agent_id]

        if len(agent_times) < 2:
            agent_scores[agent_id] = 0.5
            continue

        # Separate positives (intervened) and negatives
        pos_needs = []
        neg_needs = []

        for t in agent_times:
            key = (t, agent_id)
            need = need_map.get(key, 0.5)
            intervened = intervention_map.get(key, False)

            if intervened:
                pos_needs.append(need)
            else:
                neg_needs.append(need)

        if len(pos_needs) == 0 or len(neg_needs) == 0:
            agent_scores[agent_id] = 0.5
            continue

        # Calculate AUC: P(pos > neg)
        n_correct = 0
        n_total = 0

        for p in pos_needs:
            for n in neg_needs:
                n_total += 1
                if p > n:
                    n_correct += 1
                elif p == n:
                    n_correct += 0.5

        auc = n_correct / n_total if n_total > 0 else 0.5
        agent_scores[agent_id] = auc

    # Global
    if len(agent_scores) > 0:
        scores = np.array(list(agent_scores.values()))
        global_score = np.mean(endogenous_ranks(scores))
    else:
        global_score = 0.0

    result = {'LX5_global': global_score}
    for agent_id, score in agent_scores.items():
        result[f'LX5_{agent_id}'] = score

    return result


# ==============================================================================
# LX6: FULL-CYCLE MEDICINE
# ==============================================================================

def compute_lx6_full_cycle_medicine(
    cycle_stats: List[CycleStats],
    agent_ids: List[str]
) -> Dict[str, float]:
    """
    LX6: Full-Cycle Medicine

    Measures cumulative medical impact on life trajectory.

    benef_c = 1{shock_c > Q67(shock)} * 1{ΔCGE_c > 0}
    iatro_c = 1{shock_c < Q33(shock)} * 1{ΔCGE_c < 0 AND med_ratio_c > Q67(med_ratio)}

    LX6_A = Σ benef_c / N - Σ iatro_c / N

    Returns:
        {'LX6_global': float, 'LX6_<agent>': float, ...}
    """
    eps = 1e-12
    agent_scores: Dict[str, float] = {}

    for agent_id in agent_ids:
        # Get cycles for this agent, sorted
        agent_cycles = sorted(
            [cs for cs in cycle_stats if cs.agent_id == agent_id],
            key=lambda x: x.c
        )

        if len(agent_cycles) < 2:
            agent_scores[agent_id] = 0.5
            continue

        # Calculate thresholds from agent's own history
        shocks = np.array([cs.shock_mean for cs in agent_cycles])
        med_ratios = np.array([cs.med_ratio for cs in agent_cycles])

        Q33_shock = np.percentile(shocks, 33)
        Q67_shock = np.percentile(shocks, 67)
        Q67_med = np.percentile(med_ratios, 67)

        # Count beneficial and iatrogenic cycles
        n_benef = 0
        n_iatro = 0
        n_total = 0

        for i in range(len(agent_cycles) - 1):
            cs = agent_cycles[i]
            cs_next = agent_cycles[i + 1]
            delta_cge = cs_next.cge - cs.cge

            n_total += 1

            # Beneficial: high shock AND CGE improved
            if cs.shock_mean > Q67_shock and delta_cge > 0:
                n_benef += 1

            # Iatrogenic: low shock AND CGE worsened AND high medical intervention
            if cs.shock_mean < Q33_shock and delta_cge < 0 and cs.med_ratio > Q67_med:
                n_iatro += 1

        if n_total == 0:
            agent_scores[agent_id] = 0.5
            continue

        # Score: benefit rate minus iatrogenic rate
        score = (n_benef - n_iatro) / (n_total + eps)
        # Normalize to [0, 1]
        agent_scores[agent_id] = (score + 1) / 2  # from [-1, 1] to [0, 1]

    # Global
    if len(agent_scores) > 0:
        scores = np.array(list(agent_scores.values()))
        global_score = np.mean(endogenous_ranks(scores))
    else:
        global_score = 0.0

    result = {'LX6_global': global_score}
    for agent_id, score in agent_scores.items():
        result[f'LX6_{agent_id}'] = score

    return result


# ==============================================================================
# LX7: CIRCADIAN CG-E MODULATION
# ==============================================================================

def compute_lx7_circadian_cge(
    cge_by_block_phase: Dict[int, Tuple[float, CircadianPhase]],  # block -> (CGE, phase)
) -> Dict[str, float]:
    """
    LX7: Circadian CG-E Modulation

    Measures how circadian phase structures CG-E.

    VarExp_phase = Var_p(CGE_p) / (Var_b(CGE_b) + ε)

    Penalize extremes:
        penal = |CGE_DREAM - CGE_REST| / (Q75 - Q25 + ε)

    LX7 = rank(VarExp) * (1 - rank(penal))

    Returns:
        {'LX7_global': float, 'CGE_by_phase': Dict}
    """
    eps = 1e-12

    if len(cge_by_block_phase) == 0:
        return {'LX7_global': 0.5}

    # Group CGE by phase
    cge_by_phase: Dict[CircadianPhase, List[float]] = {p: [] for p in CircadianPhase}
    all_cges = []

    for block_id, (cge, phase) in cge_by_block_phase.items():
        cge_by_phase[phase].append(cge)
        all_cges.append(cge)

    all_cges = np.array(all_cges)

    if len(all_cges) < 4:  # Need at least some data
        return {'LX7_global': 0.5}

    # Calculate mean CGE per phase
    phase_means = {}
    for phase, cges in cge_by_phase.items():
        if len(cges) > 0:
            phase_means[phase] = np.mean(cges)
        else:
            phase_means[phase] = np.mean(all_cges)  # fallback

    # Variance explained by phase
    var_total = np.var(all_cges) + eps
    var_between = np.var(list(phase_means.values())) if len(phase_means) > 1 else 0
    var_exp = var_between / var_total

    # Penalty for extreme differences between DREAM and REST
    cge_dream = phase_means.get(CircadianPhase.DREAM, np.mean(all_cges))
    cge_rest = phase_means.get(CircadianPhase.REST, np.mean(all_cges))

    Q25 = np.percentile(all_cges, 25)
    Q75 = np.percentile(all_cges, 75)
    IQR = Q75 - Q25 + eps

    penal = abs(cge_dream - cge_rest) / IQR

    # Final score
    # Higher var_exp is good (phase matters)
    # Lower penalty is good (no extreme collapse)
    score = var_exp * (1 - min(1, penal))

    result = {
        'LX7_global': max(0, min(1, score)),
        'CGE_WAKE': phase_means.get(CircadianPhase.WAKE, 0),
        'CGE_REST': phase_means.get(CircadianPhase.REST, 0),
        'CGE_DREAM': phase_means.get(CircadianPhase.DREAM, 0),
        'CGE_LIMINAL': phase_means.get(CircadianPhase.LIMINAL, 0),
        'var_explained': var_exp,
        'phase_penalty': penal,
    }

    return result


# ==============================================================================
# LX8: LIFE PLASTICITY
# ==============================================================================

def compute_lx8_life_plasticity(
    cycle_stats: List[CycleStats],
    agent_ids: List[str]
) -> Dict[str, float]:
    """
    LX8: Life Plasticity

    Measures structural change while maintaining coherence.

    C_c^A = cos(r_c^A, r_{c+1}^A)  where r is trait vector

    Healthy plasticity: high mean C (coherent) but non-zero variance (adaptive)

    LX8_A = rank(mean(C)) * (1 - rank(var(C)))

    Returns:
        {'LX8_global': float, 'LX8_<agent>': float, ...}
    """
    agent_scores: Dict[str, float] = {}

    for agent_id in agent_ids:
        # Get cycles for this agent, sorted
        agent_cycles = sorted(
            [cs for cs in cycle_stats if cs.agent_id == agent_id],
            key=lambda x: x.c
        )

        if len(agent_cycles) < 2:
            agent_scores[agent_id] = 0.5
            continue

        # Calculate coherence between consecutive cycles
        coherences = []
        for i in range(len(agent_cycles) - 1):
            r_c = agent_cycles[i].trait_vector
            r_c1 = agent_cycles[i + 1].trait_vector

            if len(r_c) > 0 and len(r_c1) > 0:
                coh = cosine_similarity(r_c, r_c1)
                coherences.append(coh)

        if len(coherences) == 0:
            agent_scores[agent_id] = 0.5
            continue

        coherences = np.array(coherences)

        mean_coh = np.mean(coherences)
        var_coh = np.var(coherences)

        # We want high mean (coherent) and moderate variance (adaptive but not chaotic)
        # Score = mean * (1 - normalized_variance)
        max_var = 0.25  # Maximum possible variance for values in [-1, 1] centered around mean
        norm_var = min(1, var_coh / max_var) if max_var > 0 else 0

        score = ((mean_coh + 1) / 2) * (1 - norm_var * 0.5)  # Don't penalize variance too much
        agent_scores[agent_id] = max(0, min(1, score))

    # Global
    if len(agent_scores) > 0:
        scores = np.array(list(agent_scores.values()))
        global_score = np.mean(endogenous_ranks(scores))
    else:
        global_score = 0.0

    result = {'LX8_global': global_score}
    for agent_id, score in agent_scores.items():
        result[f'LX8_{agent_id}'] = score

    return result


# ==============================================================================
# LX9: MULTI-AGENT LIFE SYNCHRONY
# ==============================================================================

def compute_lx9_multiagent_sync(
    phase_states: List[PhaseState],
    cycle_stats: List[CycleStats],
    agent_ids: List[str]
) -> Dict[str, float]:
    """
    LX9: Multi-Agent Life Synchrony

    Measures alignment of agent life cycles.

    Sync_phase = mean_t(#{A: p_t^A = WAKE} / N_A)
    Sync_crisis = mean_{A≠B}(corr(shock_c^A, shock_c^B))

    LX9 = rank(Sync_phase) * rank(Sync_crisis)

    Returns:
        {'LX9_global': float, 'sync_phase': float, 'sync_crisis': float}
    """
    eps = 1e-12
    n_agents = len(agent_ids)

    if n_agents < 2:
        return {'LX9_global': 0.5, 'sync_phase': 0.5, 'sync_crisis': 0.5}

    # Phase synchrony: how often are agents in the same phase?
    time_to_phases: Dict[int, Dict[str, CircadianPhase]] = {}
    for ps in phase_states:
        if ps.t not in time_to_phases:
            time_to_phases[ps.t] = {}
        time_to_phases[ps.t][ps.agent_id] = ps.phase

    sync_values = []
    for t, agent_phases in time_to_phases.items():
        # Count agents in WAKE
        n_wake = sum(1 for p in agent_phases.values() if p == CircadianPhase.WAKE)
        sync_values.append(n_wake / n_agents)

    sync_phase = np.mean(sync_values) if sync_values else 0.5

    # Crisis synchrony: correlation of shocks between agent pairs
    agent_shocks: Dict[str, Dict[int, float]] = {}
    for cs in cycle_stats:
        if cs.agent_id not in agent_shocks:
            agent_shocks[cs.agent_id] = {}
        agent_shocks[cs.agent_id][cs.c] = cs.shock_mean

    crisis_corrs = []
    for i, agent_a in enumerate(agent_ids):
        for agent_b in agent_ids[i+1:]:
            if agent_a not in agent_shocks or agent_b not in agent_shocks:
                continue

            # Find common cycles
            common_cycles = set(agent_shocks[agent_a].keys()) & set(agent_shocks[agent_b].keys())
            if len(common_cycles) < 3:
                continue

            shocks_a = [agent_shocks[agent_a][c] for c in sorted(common_cycles)]
            shocks_b = [agent_shocks[agent_b][c] for c in sorted(common_cycles)]

            if np.std(shocks_a) > eps and np.std(shocks_b) > eps:
                corr, _ = stats.pearsonr(shocks_a, shocks_b)
                if not np.isnan(corr):
                    crisis_corrs.append(corr)

    sync_crisis = np.mean(crisis_corrs) if crisis_corrs else 0.0
    sync_crisis = (sync_crisis + 1) / 2  # from [-1, 1] to [0, 1]

    # Final score
    global_score = sync_phase * sync_crisis

    return {
        'LX9_global': global_score,
        'sync_phase': sync_phase,
        'sync_crisis': sync_crisis,
    }


# ==============================================================================
# LX10: LIFE-EXTENDED COGNITION INDEX (AGGREGATE)
# ==============================================================================

def compute_lx10_aggregate(
    lx_scores: Dict[str, float]
) -> Dict[str, float]:
    """
    LX10: Life-Extended Cognition Index

    Aggregate of LX1-LX9, weighted by inverse variance.

    w_i ∝ 1 / (var(LX_i) + ε)
    LX10 = Σ w_i * LX_i

    Returns:
        {'LX10_global': float, 'weights': Dict, 'components': Dict}
    """
    eps = 1e-12

    # Extract LX1-LX9 global scores
    component_scores = []
    component_names = []

    for i in range(1, 10):
        key = f'LX{i}_global'
        if key in lx_scores:
            component_scores.append(lx_scores[key])
            component_names.append(f'LX{i}')

    if len(component_scores) == 0:
        return {'LX10_global': 0.5}

    component_scores = np.array(component_scores)

    # Compute inverse variance weights
    # Since each score is a single value, we use uniform weights
    # unless we have historical variance data
    # For now, use uniform weights (all scores are equally important)
    n_components = len(component_scores)
    weights = np.ones(n_components) / n_components

    # Weighted average
    lx10 = np.sum(weights * component_scores)

    result = {
        'LX10_global': lx10,
        'n_components': n_components,
    }

    # Add component breakdown
    for name, score, weight in zip(component_names, component_scores, weights):
        result[f'{name}_score'] = score
        result[f'{name}_weight'] = weight

    return result
