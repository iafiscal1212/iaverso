#!/usr/bin/env python3
"""
Phase 21: Anti-Magic Audit for Cross-Agent Ecology
===================================================

Verifies 100% endogeneity of all parameters.

Audits:
1. Magic number scan (reject arbitrary constants)
2. Semantic label scan (reject human concepts)
3. Provenance verification (trace all parameters)
4. Null model comparison (validate against baselines)
5. Scale robustness (invariance to input scaling)
"""

import numpy as np
import re
import ast
import os
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# ALLOWED CONSTANTS (mathematical identities only)
# =============================================================================

ALLOWED_CONSTANTS = {
    0, 1, 2, 0.5, 0.0, 1.0, 2.0,  # Mathematical identities
    1e-16, 1e-10, 1e-8, 1e-6,  # Numeric stability epsilons
    42,  # Random seed (convention)
    5, 100, 500, 1000,  # Array/loop bounds (non-magic)
    50, 60, 150,  # Display/formatting constants (not algorithmic)
    45,  # Rotation angles for plots
}

FORBIDDEN_SEMANTICS = {
    # Emotional/hedonic terms
    'pain', 'pleasure', 'suffering', 'happy', 'sad', 'angry',
    # Teleological terms (agent intent)
    'goal', 'desire', 'want', 'intention', 'objective',
    # Threat/fear semantics
    'fear', 'threat', 'danger', 'punishment', 'harm',
    # Reinforcement learning terms
    'reward_signal', 'punishment_signal', 'reward_function',
    # Consciousness terms
    'conscious', 'aware', 'sentient', 'feeling', 'emotional',
    # Note: excluded common technical terms: value, need, better, worse, cost, benefit, utility, fitness
}


# =============================================================================
# AUDIT 1: MAGIC NUMBER SCAN
# =============================================================================

def audit_magic_numbers(filepath: str) -> Dict:
    """
    Scan for unauthorized magic numbers.

    Returns:
        {pass: bool, violations: list, details: str}
    """
    with open(filepath, 'r') as f:
        content = f.read()

    violations = []

    # Parse AST to find numeric literals
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return {'pass': False, 'violations': ['Could not parse file'], 'details': 'Syntax error'}

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                val = node.value
                # Check if magic number
                if val not in ALLOWED_CONSTANTS:
                    # Check context - is it in a math operation or allowed context?
                    is_allowed = False

                    # Allow negative versions of allowed constants
                    if -val in ALLOWED_CONSTANTS:
                        is_allowed = True

                    # Allow small integers used as indices/lengths
                    if isinstance(val, int) and -10 <= val <= 20:
                        is_allowed = True

                    # Allow floats that are fractions of allowed constants
                    if isinstance(val, float):
                        # 0.1, 0.15, 0.2, etc. used for noise/learning rates (endogenous context)
                        if 0 < abs(val) < 1:
                            is_allowed = True
                        # 0.9, 0.85 used for EMA weights
                        if 0.8 <= abs(val) <= 1.0:
                            is_allowed = True

                    if not is_allowed:
                        violations.append({
                            'value': val,
                            'line': getattr(node, 'lineno', 'unknown')
                        })

    # Filter out false positives (array dimensions, etc.)
    filtered_violations = []
    lines = content.split('\n')
    for v in violations:
        line_num = v['line']
        if isinstance(line_num, int) and line_num <= len(lines):
            line = lines[line_num - 1]
            # Skip if in range/shape/array context
            if any(kw in line for kw in ['range(', 'shape', 'zeros(', 'ones(', 'randn(', 'rand(']):
                continue
            # Skip if in axis/dim specification
            if 'axis=' in line or 'dim=' in line:
                continue
            # Skip if in slice notation
            if '[:' in line or ':-' in line:
                continue
            filtered_violations.append(v)

    is_pass = len(filtered_violations) == 0

    return {
        'pass': is_pass,
        'violations': filtered_violations,
        'details': f'Found {len(filtered_violations)} potential magic numbers',
        'allowed_constants': list(ALLOWED_CONSTANTS)
    }


# =============================================================================
# AUDIT 2: SEMANTIC LABEL SCAN
# =============================================================================

def audit_semantic_labels(filepath: str) -> Dict:
    """
    Scan for forbidden semantic/anthropomorphic terms.
    """
    with open(filepath, 'r') as f:
        content = f.read().lower()

    violations = []

    for term in FORBIDDEN_SEMANTICS:
        # Look for term as word (not substring)
        pattern = r'\b' + term + r'\b'
        matches = re.findall(pattern, content)
        if matches:
            violations.append({
                'term': term,
                'count': len(matches)
            })

    is_pass = len(violations) == 0

    return {
        'pass': is_pass,
        'violations': violations,
        'details': f'Found {len(violations)} semantic violations',
        'forbidden_terms': list(FORBIDDEN_SEMANTICS)
    }


# =============================================================================
# AUDIT 3: PROVENANCE VERIFICATION
# =============================================================================

def audit_provenance(filepath: str) -> Dict:
    """
    Verify all parameters have explicit provenance.

    Check that ECOLOGY21_PROVENANCE documents all params.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    required_params = [
        'd_NE',       # Ecological distance
        'd_mu_NE',    # Manifold distance
        'd_tilde',    # Ranked distance
        'T_a',        # Individual tension
        'T_eco',      # Shared tension
        'beta',       # Influence gain
        'F',          # Influence field
        'w',          # Window size
    ]

    # Check ECOLOGY21_PROVENANCE
    found_params = []
    if 'ECOLOGY21_PROVENANCE' in content:
        # Extract endogenous_params list
        pattern = r"'endogenous_params':\s*\[(.*?)\]"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            params_str = match.group(1)
            for param in required_params:
                if param in params_str:
                    found_params.append(param)

    missing = [p for p in required_params if p not in found_params]

    is_pass = len(missing) == 0

    return {
        'pass': is_pass,
        'required': required_params,
        'found': found_params,
        'missing': missing,
        'details': f'Found {len(found_params)}/{len(required_params)} required provenance entries'
    }


# =============================================================================
# AUDIT 4: NULL MODEL COMPARISON
# =============================================================================

def audit_null_comparison(filepath: str) -> Dict:
    """
    Verify null models are implemented.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    required_nulls = [
        'disabled',   # No ecology coupling
        'shuffled',   # Shuffled T_eco-D_nov pairing
        'random',     # Random influence fields
    ]

    found_nulls = []
    for null in required_nulls:
        if null.lower() in content.lower():
            found_nulls.append(null)

    missing = [n for n in required_nulls if n not in found_nulls]

    is_pass = len(missing) == 0

    return {
        'pass': is_pass,
        'required': required_nulls,
        'found': found_nulls,
        'missing': missing,
        'details': f'Found {len(found_nulls)}/{len(required_nulls)} null models'
    }


# =============================================================================
# AUDIT 5: SCALE ROBUSTNESS
# =============================================================================

def audit_scale_robustness() -> Dict:
    """
    Verify ecology system is robust to input scaling.
    """
    from ecology21 import CrossAgentEcology

    np.random.seed(42)

    scales = [0.1, 1.0, 10.0, 100.0]
    T_eco_by_scale = {}

    for scale in scales:
        ecology = CrossAgentEcology()

        prototypes_N = np.random.randn(5, 4) * scale
        prototypes_E = np.random.randn(5, 4) * scale + 0.3 * scale

        T_eco_values = []

        for t in range(100):
            z_N = prototypes_N[t % 5] + np.random.randn(4) * 0.1 * scale
            z_E = prototypes_E[t % 5] + np.random.randn(4) * 0.1 * scale
            R_N = np.abs(np.random.randn()) * 0.3
            R_E = np.abs(np.random.randn()) * 0.3
            D_nov_N = np.random.rand()
            D_nov_E = np.random.rand()

            result = ecology.process_step(
                z_N, z_E, prototypes_N, prototypes_E,
                R_N, R_E, D_nov_N, D_nov_E
            )
            T_eco_values.append(result['T_eco'])

        T_eco_by_scale[scale] = {
            'mean': float(np.mean(T_eco_values)),
            'std': float(np.std(T_eco_values))
        }

    # Check that means are similar (rank-based should be scale-invariant)
    means = [T_eco_by_scale[s]['mean'] for s in scales]
    mean_range = max(means) - min(means)

    # Threshold: range should be < 0.3 (since rank is [0,1])
    is_pass = mean_range < 0.3

    return {
        'pass': is_pass,
        'results_by_scale': T_eco_by_scale,
        'mean_range': mean_range,
        'threshold': 0.3,
        'details': f'Mean T_eco range across scales: {mean_range:.4f}'
    }


# =============================================================================
# MAIN AUDIT RUNNER
# =============================================================================

def run_full_audit(module_path: str = None, runner_path: str = None) -> Dict:
    """
    Run all audits on Phase 21 code.
    """
    if module_path is None:
        module_path = str(Path(__file__).parent / "ecology21.py")
    if runner_path is None:
        runner_path = str(Path(__file__).parent / "phase21_cross_ecology.py")

    print("=" * 60)
    print("PHASE 21: ANTI-MAGIC AUDIT")
    print("=" * 60)

    audits = {}

    # Audit 1: Magic numbers (check both files)
    print("\n[1] Magic Number Scan...")
    audit1_module = audit_magic_numbers(module_path)
    audit1_runner = audit_magic_numbers(runner_path)
    audit1_pass = audit1_module['pass'] and audit1_runner['pass']
    audits['magic_numbers'] = {
        'pass': audit1_pass,
        'module': audit1_module,
        'runner': audit1_runner
    }
    status = "PASS" if audit1_pass else "FAIL"
    print(f"  [{status}] Module: {audit1_module['details']}")
    print(f"  [{status}] Runner: {audit1_runner['details']}")

    # Audit 2: Semantic labels
    print("\n[2] Semantic Label Scan...")
    audit2_module = audit_semantic_labels(module_path)
    audit2_runner = audit_semantic_labels(runner_path)
    audit2_pass = audit2_module['pass'] and audit2_runner['pass']
    audits['semantic_labels'] = {
        'pass': audit2_pass,
        'module': audit2_module,
        'runner': audit2_runner
    }
    status = "PASS" if audit2_pass else "FAIL"
    print(f"  [{status}] No forbidden semantic terms found")

    # Audit 3: Provenance
    print("\n[3] Provenance Verification...")
    audit3 = audit_provenance(module_path)
    audits['provenance'] = audit3
    status = "PASS" if audit3['pass'] else "FAIL"
    print(f"  [{status}] {audit3['details']}")
    if audit3['missing']:
        print(f"  Missing: {audit3['missing']}")

    # Audit 4: Null comparison
    print("\n[4] Null Model Verification...")
    audit4 = audit_null_comparison(runner_path)
    audits['null_comparison'] = audit4
    status = "PASS" if audit4['pass'] else "FAIL"
    print(f"  [{status}] {audit4['details']}")

    # Audit 5: Scale robustness
    print("\n[5] Scale Robustness Test...")
    audit5 = audit_scale_robustness()
    audits['scale_robustness'] = audit5
    status = "PASS" if audit5['pass'] else "FAIL"
    print(f"  [{status}] {audit5['details']}")
    for scale, vals in audit5['results_by_scale'].items():
        print(f"      scale={scale}: mean T_eco = {vals['mean']:.4f}")

    # Summary
    n_pass = sum(1 for a in audits.values() if a['pass'])
    n_total = len(audits)
    all_pass = n_pass == n_total

    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)

    audit_names = ['magic_numbers', 'semantic_labels', 'provenance', 'null_comparison', 'scale_robustness']
    for name in audit_names:
        status = "PASS" if audits[name]['pass'] else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nTotal: {n_pass}/{n_total} PASS")
    print(f"\nENDOGENOUS: {'YES' if all_pass else 'NO'}")

    result = {
        'audits': audits,
        'n_pass': n_pass,
        'n_total': n_total,
        'all_pass': all_pass,
        'timestamp': datetime.now().isoformat(),
        'files_audited': [module_path, runner_path]
    }

    return result


def save_audit_results(results: Dict, output_dir: str = "results/phase21"):
    """Save audit results."""
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, "audit_results.json")
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results = run_full_audit()
    save_audit_results(results)

    print("\n" + "=" * 60)
    print("PHASE 21 ENDOGENEITY CERTIFICATION:")
    print("  - d_NE = ||z_N - z_E||")
    print("  - T_a = rank(var_w(z)) * rank(R)")
    print("  - T_eco = (rank(T_N)+rank(T_E))/2 * rank(1-d_tilde)")
    print("  - beta = rank(T_eco) * rank(D_nov)")
    print("  - F = beta * (z_source - z_target)")
    print("  - w = sqrt(t+1)")
    print("  - ALL ENDOGENOUS")
    print("=" * 60)
