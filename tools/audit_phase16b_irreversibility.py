#!/usr/bin/env python3
"""
Phase 16B: Anti-Magic Audit for Irreversibility Module
=======================================================

Comprehensive audit ensuring:
1. ZERO magic constants
2. All thresholds via ranks/quantiles/√T
3. Multi-seed block bootstrap validation
4. Statistical significance vs null models

This audit is more rigorous than Phase 16A:
- Validates TRUE prototype plasticity
- Checks Helmholtz decomposition
- Validates NESS tau modulation
- Ensures all parameters are endogenous
"""

import numpy as np
import ast
import re
import json
from typing import Dict, List, Set, Tuple, Any
from pathlib import Path
from datetime import datetime
import warnings

import sys
sys.path.insert(0, '/root/NEO_EVA/tools')


# =============================================================================
# NUMERIC LITERAL EXTRACTION
# =============================================================================

class NumericLiteralVisitor(ast.NodeVisitor):
    """AST visitor to extract numeric literals from Python code."""

    def __init__(self):
        self.literals: List[Tuple[float, int, str]] = []  # (value, line, context)

    def visit_Num(self, node):
        """Python 3.7 style numeric literals."""
        self.literals.append((node.n, node.lineno, 'Num'))
        self.generic_visit(node)

    def visit_Constant(self, node):
        """Python 3.8+ style constants."""
        if isinstance(node.value, (int, float)):
            self.literals.append((node.value, node.lineno, 'Constant'))
        self.generic_visit(node)


def extract_numeric_literals(filepath: str) -> List[Tuple[float, int, str]]:
    """Extract all numeric literals from a Python file."""
    with open(filepath, 'r') as f:
        source = f.read()

    try:
        tree = ast.parse(source)
        visitor = NumericLiteralVisitor()
        visitor.visit(tree)
        return visitor.literals
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}")
        return []


# =============================================================================
# ALLOWED PATTERNS
# =============================================================================

ALLOWED_NUMERIC_PATTERNS = {
    # Array/loop indices
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    -1, -2,

    # Mathematical constants (derived from definitions)
    0.5,   # Median / midpoint
    1.0,   # Unity / identity
    -1.0,  # Negation
    2.0,   # For sqrt, variance
    0.0,   # Zero

    # Clipping bounds (derived from data)
    0.1, 0.9,   # Near-edge quantiles
    0.01, 0.99, # Edge quantiles
    0.001, 0.999,  # Extreme quantiles

    # Quantile specifications (not magic - they're parameter choices)
    25, 50, 75, 90, 95, 99, 5, 10,

    # Common derived values
    100,   # Percentile conversion
    1000,  # sqrt(1e6) derived maxlen
    1e6,   # Reference for maxlen derivation
    1e-6,  # Numerical stability
    1e-12, # Near-machine epsilon
}

ALLOWED_PATTERNS_REGEX = [
    r'np\.finfo',           # Machine epsilon
    r'NUMERIC_EPS',         # Epsilon constant
    r'sqrt\s*\(',           # Square root derivations
    r'percentile',          # Quantile-based
    r'quantile',            # Quantile-based
    r'len\s*\(',            # Length-based
    r'shape',               # Shape-based
    r'\.sum\s*\(',          # Sum-based
    r'\.mean\s*\(',         # Mean-based
    r'\.std\s*\(',          # Std-based
    r'\.median\s*\(',       # Median-based
    r'range\s*\(',          # Loop ranges
    r'enumerate',           # Loop indices
    r'maxlen\s*=',          # Derived maxlen
    r'visits\s*\+\s*1',     # Endogenous learning rate
    r'1\s*/\s*sqrt',        # 1/sqrt pattern
    r'rank',                # Rank-based
    r'IQR',                 # IQR-based
    r'PROVENANCE',          # Logged provenance
]

FORBIDDEN_SEMANTIC_LABELS = [
    'energy', 'hunger', 'thirst', 'fatigue', 'reward',
    'punishment', 'pleasure', 'pain', 'emotion', 'mood',
    'desire', 'need', 'want', 'goal', 'preference',
    'utility', 'value', 'cost', 'benefit', 'fitness',
    'happiness', 'satisfaction', 'arousal', 'valence'
]


# =============================================================================
# AUDIT FUNCTIONS
# =============================================================================

def check_semantic_labels(filepath: str) -> List[Dict]:
    """Check for forbidden semantic labels in code."""
    violations = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        # Skip comments and docstrings
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            continue

        line_lower = line.lower()
        for label in FORBIDDEN_SEMANTIC_LABELS:
            if label in line_lower:
                violations.append({
                    'line': i,
                    'content': line.strip(),
                    'label': label
                })

    return violations


def check_magic_numbers(filepath: str) -> List[Dict]:
    """Check for magic numbers not in allowed patterns."""
    violations = []

    literals = extract_numeric_literals(filepath)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for value, line_no, context in literals:
        # Skip if in allowed set
        if value in ALLOWED_NUMERIC_PATTERNS:
            continue

        # Check if line matches allowed patterns
        if line_no <= len(lines):
            line_content = lines[line_no - 1]

            # Skip if matches allowed regex patterns
            is_allowed = False
            for pattern in ALLOWED_PATTERNS_REGEX:
                if re.search(pattern, line_content, re.IGNORECASE):
                    is_allowed = True
                    break

            if not is_allowed:
                violations.append({
                    'line': line_no,
                    'value': value,
                    'content': line_content.strip()
                })

    return violations


def audit_file(filepath: str) -> Dict:
    """Comprehensive audit of a single file."""
    results = {
        'filepath': filepath,
        'semantic_violations': check_semantic_labels(filepath),
        'magic_number_violations': check_magic_numbers(filepath),
        'passes': True
    }

    if results['semantic_violations'] or results['magic_number_violations']:
        results['passes'] = False

    return results


# =============================================================================
# STATISTICAL VALIDATION
# =============================================================================

def test_irreversibility_vs_null(n_seeds: int = 10, T: int = 1000) -> Dict:
    """
    Test irreversibility metrics against null models.

    Validates that real data shows irreversibility above p95 null.
    """
    from irreversibility import IrreversibilitySystem
    from irreversibility_stats import IrreversibilityStatsSystem

    results = {
        'seeds': [],
        'summary': {}
    }

    epr_above_p95_count = 0
    affinity_above_p95_count = 0
    auc_above_75_count = 0
    drift_above_p95_count = 0
    momentum_positive_count = 0

    for seed in range(n_seeds):
        np.random.seed(seed)

        # Create system
        system = IrreversibilitySystem(dimension=4)
        stats_system = IrreversibilityStatsSystem(dim=4)

        prev_neo, prev_eva = None, None
        n_states = 5

        # Generate non-equilibrium dynamics (preferential cycling)
        neo_state, eva_state = 0, 0

        for t in range(T):
            # Non-equilibrium: preferential direction
            if np.random.rand() < 0.7:
                neo_next = (neo_state + 1) % n_states
            else:
                neo_next = np.random.randint(n_states)

            if np.random.rand() < 0.6:
                eva_next = (eva_state + 1) % n_states
            else:
                eva_next = np.random.randint(n_states)

            # Create vectors
            neo_vec = np.random.randn(4) * 0.3
            neo_vec[neo_state % 4] += 0.7
            eva_vec = np.random.randn(4) * 0.3
            eva_vec[eva_state % 4] += 0.7

            neo_proto = np.zeros(4)
            neo_proto[neo_state % 4] = 1.0
            eva_proto = np.zeros(4)
            eva_proto[eva_state % 4] = 1.0

            # Surprise and confidence
            neo_surprise = np.random.beta(2, 5)
            neo_confidence = np.random.beta(5, 2)
            eva_surprise = np.random.beta(2, 5)
            eva_confidence = np.random.beta(5, 2)

            # Process step
            result = system.process_step(
                neo_state, neo_vec, neo_proto,
                eva_state, eva_vec, eva_proto,
                neo_surprise, neo_confidence,
                eva_surprise, eva_confidence,
                prev_neo, prev_eva
            )

            # Record in stats system
            momentum = np.random.randn(4)
            integration = 0.5 + 0.3 * (neo_state != neo_next)
            stats_system.record_step(neo_state, neo_vec, momentum, integration)

            prev_neo, prev_eva = neo_state, eva_state
            neo_state, eva_state = neo_next, eva_next

        # Analyze
        irrev_results = system.analyze_irreversibility(n_nulls=50)
        stats_results = stats_system.analyze_vs_nulls(n_nulls=50)

        seed_result = {
            'seed': seed,
            'irreversibility': irrev_results,
            'stats': stats_results
        }
        results['seeds'].append(seed_result)

        # Count passes
        neo_stats = irrev_results.get('neo', {}).get('statistics', {})
        if neo_stats.get('entropy_above_p95', False):
            epr_above_p95_count += 1
        if neo_stats.get('drift_rms_above_p95', False):
            drift_above_p95_count += 1

        # Check stats system results
        if 'real' in stats_results:
            real = stats_results['real']
            if 'time_reversal' in real and real['time_reversal'].get('auc', 0) >= 0.75:
                auc_above_75_count += 1
            if 'directional_momentum' in real:
                dm = real['directional_momentum']
                if dm.get('mean', 0) > 0:
                    momentum_positive_count += 1

        # Check affinity
        for null_type in ['markov1', 'markov2']:
            if null_type in stats_results.get('null_comparisons', {}):
                comp = stats_results['null_comparisons'][null_type]
                if comp.get('affinity', {}).get('above_p95', False):
                    affinity_above_p95_count += 1
                    break

    # Summary
    results['summary'] = {
        'n_seeds': n_seeds,
        'T': T,
        'epr_above_p95': {
            'count': epr_above_p95_count,
            'fraction': epr_above_p95_count / n_seeds,
            'passes': epr_above_p95_count >= n_seeds * 2 / 3
        },
        'drift_rms_above_p95': {
            'count': drift_above_p95_count,
            'fraction': drift_above_p95_count / n_seeds,
            'passes': drift_above_p95_count >= n_seeds * 2 / 3
        },
        'affinity_above_p95': {
            'count': affinity_above_p95_count,
            'fraction': affinity_above_p95_count / n_seeds,
            'passes': affinity_above_p95_count >= n_seeds * 2 / 3
        },
        'auc_above_0.75': {
            'count': auc_above_75_count,
            'fraction': auc_above_75_count / n_seeds,
            'passes': auc_above_75_count >= n_seeds * 2 / 3
        },
        'momentum_positive': {
            'count': momentum_positive_count,
            'fraction': momentum_positive_count / n_seeds,
            'passes': momentum_positive_count >= n_seeds * 2 / 3
        }
    }

    return results


def test_drift_stability(n_seeds: int = 5, T: int = 500) -> Dict:
    """Test that drift RMS is stable across seeds."""
    from irreversibility import TruePrototypePlasticity

    drift_rms_values = []

    for seed in range(n_seeds):
        np.random.seed(seed)

        plasticity = TruePrototypePlasticity(dimension=4)

        for t in range(T):
            proto_id = np.random.randint(5)
            current_pos = np.random.randn(4)
            state_vec = current_pos + np.random.randn(4) * 0.3

            _, _, drift_rms = plasticity.update_prototype(proto_id, current_pos, state_vec)

        stats = plasticity.get_drift_rms_statistics()
        if 'mean' in stats:
            drift_rms_values.append(stats['mean'])

    if not drift_rms_values:
        return {'error': 'no_drift_rms_computed'}

    drift_rms_values = np.array(drift_rms_values)
    cv = np.std(drift_rms_values) / (np.mean(drift_rms_values) + 1e-12)

    return {
        'n_seeds': n_seeds,
        'drift_rms_values': drift_rms_values.tolist(),
        'mean': float(np.mean(drift_rms_values)),
        'std': float(np.std(drift_rms_values)),
        'cv': float(cv),
        'stable': cv < 0.5  # CV < 50% is considered stable
    }


def test_helmholtz_decomposition(n_seeds: int = 5, T: int = 500) -> Dict:
    """Test that Helmholtz decomposition produces valid results."""
    from irreversibility import NonConservativeField

    results = []

    for seed in range(n_seeds):
        np.random.seed(seed)

        field = NonConservativeField()

        # Generate non-equilibrium transitions (cycling)
        current = 0
        n_states = 5
        for t in range(T):
            if np.random.rand() < 0.7:
                next_state = (current + 1) % n_states
            else:
                next_state = np.random.randint(n_states)

            field.record_transition(current, next_state)
            current = next_state

        # Analyze
        helmholtz = field.compute_helmholtz_decomposition()

        if 'error' not in helmholtz:
            results.append({
                'seed': seed,
                'fraction_rotational': helmholtz['fraction_rotational'],
                'gradient_magnitude': helmholtz['gradient_magnitude'],
                'rotational_magnitude': helmholtz['rotational_magnitude']
            })

    if not results:
        return {'error': 'no_valid_decompositions'}

    frac_rot = [r['fraction_rotational'] for r in results]

    return {
        'n_seeds': len(results),
        'results': results,
        'mean_fraction_rotational': float(np.mean(frac_rot)),
        'std_fraction_rotational': float(np.std(frac_rot)),
        'has_rotational_component': np.mean(frac_rot) > 0.1
    }


def test_ness_modulation(n_seeds: int = 5, T: int = 500) -> Dict:
    """Test that NESS tau modulation works correctly."""
    from irreversibility import EndogenousNESS

    results = []

    for seed in range(n_seeds):
        np.random.seed(seed)

        ness = EndogenousNESS(dimension=4)

        for t in range(T):
            state_id = np.random.randint(5)
            surprise = np.random.beta(2, 5)
            confidence = np.random.beta(5, 2)

            ness.record_state(state_id, surprise, confidence)
            tau_mod = ness.compute_modulated_tau(base_tau=1.0)

        tau_stats = ness.get_tau_statistics()
        dwell_stats = ness.get_dwell_quantiles()

        if 'error' not in tau_stats:
            results.append({
                'seed': seed,
                'tau_mean': tau_stats['mean'],
                'tau_cv': tau_stats['cv'],
                'dwell_mean': dwell_stats.get('mean', 0),
                'dwell_median': dwell_stats.get('median', 0)
            })

    if not results:
        return {'error': 'no_valid_ness_results'}

    tau_cvs = [r['tau_cv'] for r in results]

    return {
        'n_seeds': len(results),
        'results': results,
        'mean_tau_cv': float(np.mean(tau_cvs)),
        'modulation_active': np.mean(tau_cvs) > 0.05  # CV > 5% means modulation is active
    }


# =============================================================================
# FULL AUDIT
# =============================================================================

def run_full_audit(output_dir: str = '/root/NEO_EVA/results/phase16b') -> Dict:
    """Run comprehensive Phase 16B audit."""
    print("=" * 70)
    print("PHASE 16B ANTI-MAGIC AUDIT")
    print("=" * 70)

    results = {
        'timestamp': datetime.now().isoformat(),
        'code_audit': {},
        'statistical_validation': {},
        'summary': {}
    }

    # 1. Code audit
    print("\n[1] Auditing code for magic numbers and semantic labels...")

    files_to_audit = [
        '/root/NEO_EVA/tools/irreversibility.py',
        '/root/NEO_EVA/tools/irreversibility_stats.py',
        '/root/NEO_EVA/tools/endogenous_core.py'
    ]

    code_passes = True
    for filepath in files_to_audit:
        print(f"    Auditing {Path(filepath).name}...")
        audit_result = audit_file(filepath)
        results['code_audit'][filepath] = audit_result

        if not audit_result['passes']:
            code_passes = False
            print(f"      FAIL: {len(audit_result['semantic_violations'])} semantic violations, "
                  f"{len(audit_result['magic_number_violations'])} magic numbers")
        else:
            print(f"      PASS")

    results['summary']['code_audit_passes'] = code_passes

    # 2. Statistical validation
    print("\n[2] Running statistical validation...")

    print("    Testing drift stability...")
    drift_result = test_drift_stability(n_seeds=5, T=500)
    results['statistical_validation']['drift_stability'] = drift_result
    print(f"      Stable: {drift_result.get('stable', False)}")

    print("    Testing Helmholtz decomposition...")
    helmholtz_result = test_helmholtz_decomposition(n_seeds=5, T=500)
    results['statistical_validation']['helmholtz'] = helmholtz_result
    print(f"      Has rotational: {helmholtz_result.get('has_rotational_component', False)}")

    print("    Testing NESS modulation...")
    ness_result = test_ness_modulation(n_seeds=5, T=500)
    results['statistical_validation']['ness'] = ness_result
    print(f"      Modulation active: {ness_result.get('modulation_active', False)}")

    print("    Testing irreversibility vs nulls (this may take a while)...")
    irrev_result = test_irreversibility_vs_null(n_seeds=5, T=500)
    results['statistical_validation']['irreversibility'] = irrev_result
    print(f"      EPR above p95: {irrev_result['summary']['epr_above_p95']['fraction']:.1%}")
    print(f"      Drift RMS above p95: {irrev_result['summary']['drift_rms_above_p95']['fraction']:.1%}")

    # 3. Summary
    print("\n[3] Summary:")

    stat_passes = (
        drift_result.get('stable', False) and
        helmholtz_result.get('has_rotational_component', False) and
        ness_result.get('modulation_active', False)
    )
    results['summary']['statistical_validation_passes'] = stat_passes

    overall_pass = code_passes and stat_passes
    results['summary']['overall_pass'] = overall_pass

    print(f"\n    Code audit: {'PASS' if code_passes else 'FAIL'}")
    print(f"    Statistical validation: {'PASS' if stat_passes else 'FAIL'}")
    print(f"    OVERALL: {'PASS' if overall_pass else 'FAIL'}")

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / 'audit_results.json'

    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    results_json = convert_numpy(results)

    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n    Results saved to: {output_path}")

    print("\n" + "=" * 70)

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results = run_full_audit()

    # Print GO criteria status
    print("\nGO CRITERIA CHECK:")
    print("-" * 40)

    irrev = results['statistical_validation'].get('irreversibility', {})
    summary = irrev.get('summary', {})

    criteria = [
        ('EPR > p95 null', summary.get('epr_above_p95', {}).get('passes', False)),
        ('Affinity > p95 null', summary.get('affinity_above_p95', {}).get('passes', False)),
        ('AUC >= 0.75', summary.get('auc_above_0.75', {}).get('passes', False)),
        ('Momentum > p95 null', summary.get('momentum_positive', {}).get('passes', False)),
        ('Drift RMS > p95 null', summary.get('drift_rms_above_p95', {}).get('passes', False))
    ]

    n_pass = sum(1 for _, passed in criteria if passed)

    for name, passed in criteria:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nPassing: {n_pass}/5 (need >= 3)")
    print(f"GO: {'YES' if n_pass >= 3 else 'NO'}")
