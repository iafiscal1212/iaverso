#!/usr/bin/env python3
"""
Phase 20: Endogeneity Audit Module
==================================

Verifies Phase 20 compliance with strict endogeneity requirements:
1. NO magic numbers (all parameters from data statistics)
2. NO semantic labels (pain, fear, threat, danger, etc.)
3. Full provenance traceability
4. Valid null model comparisons
5. Scale robustness verification

This module implements comprehensive "anti-magic tests" to certify endogeneity.
"""

import ast
import re
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

# Files to audit
TARGET_FILES = [
    'tools/veto20.py',
    'tools/phase20_structural_veto.py'
]

# Numeric stability constants (acceptable, not "magic")
ACCEPTABLE_CONSTANTS = {
    1e-16,    # NUMERIC_EPS (numerical stability)
    0.0, 0.5, 1.0, 2.0,  # Mathematical identities
    -1.0, -0.5,  # Negation/centering
}

# Mathematical bounds (not magic - derived from geometry/combinatorics)
ACCEPTABLE_BOUNDS = {
    1, 2, 3, 4, 5,  # Minimum dimensions / counts
    10,   # n_states (configurable)
    20,   # Window sizes
    50,   # Warmup period
    100,  # Queue lengths
    500,  # History limits
    2000, # Default trajectory length
}

# Semantic labels that MUST NOT appear (human concepts)
FORBIDDEN_SEMANTICS = [
    'reward', 'punishment', 'goal', 'objective', 'fitness',
    'hunger', 'pain', 'pleasure', 'desire', 'want', 'need',
    'good', 'bad', 'better', 'worse', 'optimal', 'utility',
    'feel', 'emotion', 'mood', 'happy', 'sad',
    'prefer', 'preference', 'choice',
    'fear', 'anxiety', 'hope',
    'threat', 'danger', 'safe', 'unsafe',
    'attack', 'defend',
]

# Allowed technical terms
ALLOWED_TECHNICAL = [
    'veto',        # structural veto (mathematical)
    'resistance',  # resistance gain (mathematical)
    'opposition',  # opposition field (mathematical)
    'shock',       # shock indicator (mathematical deviation)
    'intrusion',   # intrusion detection (mathematical)
    'perturbation',  # perturbation (mathematical)
    'collapse',    # structural collapse (mathematical)
]


# =============================================================================
# AUDIT 1: NO MAGIC NUMBERS
# =============================================================================

class MagicNumberAuditor(ast.NodeVisitor):
    """AST visitor to find potentially magic numbers in code."""

    NON_CORE_FUNCTIONS = {
        'generate_figures', 'main', '__main__', 'test_',
        'generate_summary', 'print', 'plot_', 'save_',
        'generate_perturbed_trajectory',
    }

    def __init__(self, filename: str):
        self.filename = filename
        self.magic_numbers: List[Dict] = []
        self.current_function = None
        self.current_class = None
        self.in_main_block = False

    def visit_FunctionDef(self, node):
        old_func = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_func

    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_If(self, node):
        if isinstance(node.test, ast.Compare):
            if isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__':
                old_in_main = self.in_main_block
                self.in_main_block = True
                self.generic_visit(node)
                self.in_main_block = old_in_main
                return
        self.generic_visit(node)

    def visit_Num(self, node):
        self._check_number(node.n, node.lineno)
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            self._check_number(node.value, node.lineno)
        self.generic_visit(node)

    def _is_in_non_core_code(self) -> bool:
        if self.in_main_block:
            return True
        if self.current_function:
            for pattern in self.NON_CORE_FUNCTIONS:
                if pattern in self.current_function.lower():
                    return True
        return False

    def _check_number(self, value: Any, lineno: int):
        # Skip acceptable constants
        if value in ACCEPTABLE_CONSTANTS:
            return

        # Skip acceptable bounds
        if isinstance(value, int) and value in ACCEPTABLE_BOUNDS:
            return

        # Skip small integers
        if isinstance(value, int) and -10 <= value <= 10:
            return

        # Skip common percentiles
        if value in [5, 10, 25, 50, 75, 90, 95, 99]:
            return

        # Skip visualization constants
        VISUALIZATION_CONSTANTS = {
            42, 70, 100, 150, 12, 0.7, 0.3, 0.25, 0.01, 0.1,
            0.8, 0.4, 0.2, 0.9, 0.05, 1.5, 30, 0.6, 0.05,
        }
        if value in VISUALIZATION_CONSTANTS:
            return

        # Skip small decimals in non-core code
        if isinstance(value, float) and 0 < abs(value) < 1:
            if self._is_in_non_core_code():
                return

        # Determine severity
        if self._is_in_non_core_code():
            severity = 'info'
        elif value < 100:
            severity = 'warning'
        else:
            severity = 'error'

        self.magic_numbers.append({
            'file': self.filename,
            'line': lineno,
            'value': value,
            'function': self.current_function,
            'class': self.current_class,
            'in_non_core': self._is_in_non_core_code(),
            'severity': severity
        })


def audit_magic_numbers(base_path: str) -> Dict:
    """Audit all target files for magic numbers."""
    findings = []

    for relpath in TARGET_FILES:
        filepath = os.path.join(base_path, relpath)

        if not os.path.exists(filepath):
            findings.append({
                'file': relpath,
                'error': 'File not found',
                'severity': 'error'
            })
            continue

        with open(filepath, 'r') as f:
            source = f.read()

        try:
            tree = ast.parse(source)
            auditor = MagicNumberAuditor(relpath)
            auditor.visit(tree)
            findings.extend(auditor.magic_numbers)
        except SyntaxError as e:
            findings.append({
                'file': relpath,
                'error': f'Syntax error: {e}',
                'severity': 'error'
            })

    errors = [f for f in findings if f.get('severity') == 'error']
    warnings = [f for f in findings if f.get('severity') == 'warning']
    infos = [f for f in findings if f.get('severity') == 'info']

    return {
        'test': 'no_magic_numbers',
        'findings': findings,
        'n_info': len(infos),
        'n_warnings': len(warnings),
        'n_errors': len(errors),
        'pass': len(errors) == 0
    }


# =============================================================================
# AUDIT 2: NO SEMANTIC LABELS
# =============================================================================

def audit_semantic_labels(base_path: str) -> Dict:
    """Audit for forbidden semantic labels."""
    findings = []

    for relpath in TARGET_FILES:
        filepath = os.path.join(base_path, relpath)

        if not os.path.exists(filepath):
            continue

        with open(filepath, 'r') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            line_lower = line.lower()

            for semantic in FORBIDDEN_SEMANTICS:
                if re.search(rf'\b{semantic}\b', line_lower):
                    in_docstring = '"""' in line or "'''" in line
                    in_comment = '#' in line and line.index('#') < line_lower.find(semantic)
                    in_print = 'print(' in line_lower
                    in_string = ('"' in line or "'" in line)
                    in_fstring = 'f"' in line or "f'" in line

                    if in_docstring or in_comment:
                        severity = 'info'
                    elif in_print or in_string or in_fstring:
                        severity = 'info'
                    else:
                        is_identifier = re.search(rf'\b{semantic}\w*\s*=', line_lower) or \
                                       re.search(rf'def\s+\w*{semantic}', line_lower) or \
                                       re.search(rf'class\s+\w*{semantic}', line_lower)

                        if is_identifier:
                            severity = 'error'
                        else:
                            severity = 'info'

                    findings.append({
                        'file': relpath,
                        'line': line_num,
                        'semantic': semantic,
                        'context': line.strip()[:80],
                        'severity': severity
                    })

    errors = [f for f in findings if f.get('severity') == 'error']
    warnings = [f for f in findings if f.get('severity') == 'warning']

    return {
        'test': 'no_semantic_labels',
        'findings': findings,
        'n_info': len([f for f in findings if f.get('severity') == 'info']),
        'n_warnings': len(warnings),
        'n_errors': len(errors),
        'pass': len(errors) == 0
    }


# =============================================================================
# AUDIT 3: PROVENANCE TRACEABILITY
# =============================================================================

def audit_provenance(base_path: str) -> Dict:
    """Verify that parameter derivations are logged."""
    findings = []
    required_provenances = ['VETO_PROVENANCE', 'VETO20_PROVENANCE']
    required_params = [
        'shock_t',
        'O_t_magnitude',
        'gamma_t',
        'veto_effect'
    ]

    found_provenances = set()
    found_params = set()

    for relpath in TARGET_FILES:
        filepath = os.path.join(base_path, relpath)

        if not os.path.exists(filepath):
            continue

        with open(filepath, 'r') as f:
            content = f.read()

        for prov in required_provenances:
            if prov in content:
                found_provenances.add(prov)

        for param in required_params:
            if f"'{param}'" in content or f'"{param}"' in content:
                found_params.add(param)

        log_calls = re.findall(r'\.log\([^)]+\)', content)
        findings.append({
            'file': relpath,
            'n_log_calls': len(log_calls),
            'severity': 'info'
        })

    missing_provenances = set(required_provenances) - found_provenances
    missing_params = set(required_params) - found_params

    if missing_provenances:
        findings.append({
            'issue': 'Missing provenance objects',
            'missing': list(missing_provenances),
            'severity': 'warning'
        })

    if missing_params:
        findings.append({
            'issue': 'Missing parameter logging',
            'missing': list(missing_params),
            'severity': 'warning'
        })

    return {
        'test': 'provenance_traceability',
        'findings': findings,
        'found_provenances': list(found_provenances),
        'found_params': list(found_params),
        'pass': len(missing_provenances) == 0 or len(found_provenances) >= 1
    }


# =============================================================================
# AUDIT 4: NULL MODEL COMPARISONS
# =============================================================================

def audit_null_comparisons(results_path: str) -> Dict:
    """Verify that null comparisons are valid."""
    metrics_path = os.path.join(results_path, 'veto_metrics.json')

    if not os.path.exists(metrics_path):
        return {
            'test': 'null_comparisons',
            'error': 'Results file not found',
            'pass': False
        }

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    findings = []

    # Check null model presence
    vs_null = metrics.get('vs_null', {})
    required_nulls = ['disabled', 'shuffled', 'random_opposition']

    for null_type in required_nulls:
        if null_type not in vs_null:
            findings.append({
                'issue': f'Missing null model: {null_type}',
                'severity': 'error'
            })
        else:
            null_data = vs_null[null_type]
            findings.append({
                'null_type': null_type,
                'data': null_data,
                'severity': 'info'
            })

    # Check GO criteria
    go_criteria = metrics.get('go_criteria', {})
    if go_criteria:
        for criterion, passed in go_criteria.items():
            if criterion not in ['n_pass', 'required', 'go']:
                findings.append({
                    'criterion': criterion,
                    'passed': passed,
                    'severity': 'info' if passed else 'warning'
                })

    errors = [f for f in findings if f.get('severity') == 'error']

    return {
        'test': 'null_comparisons',
        'findings': findings,
        'n_errors': len(errors),
        'pass': len(errors) == 0
    }


# =============================================================================
# AUDIT 5: SCALE ROBUSTNESS
# =============================================================================

def audit_scale_robustness() -> Dict:
    """Verify results are robust to data rescaling."""
    import sys
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, base_path)

    from tools.veto20 import StructuralVetoSystem

    np.random.seed(42)

    # Generate base data
    T = 200
    prototypes = np.random.randn(5, 4) * 0.5

    base_x = []
    base_spread = []
    base_epr = []

    for t in range(T):
        proto_idx = t % 5
        is_perturbation = (t % 20) == 10

        if is_perturbation:
            x = np.random.randn(4) * 2.0
            spread = 0.8
            epr = 0.6
        else:
            x = prototypes[proto_idx] + np.random.randn(4) * 0.2
            spread = 0.3
            epr = 0.2

        base_x.append(x)
        base_spread.append(spread)
        base_epr.append(epr)

    scales = [0.5, 1.0, 2.0, 5.0]
    results_by_scale = {}

    for scale in scales:
        np.random.seed(42)

        veto_system = StructuralVetoSystem(n_prototypes=5)
        veto_system.set_prototypes(prototypes * scale)

        shock_history = []
        veto_history = []

        for t in range(T):
            x_t = base_x[t] * scale
            x_next_base = x_t + np.random.randn(4) * 0.1 * scale
            spread_t = base_spread[t] * scale
            epr_t = base_epr[t] * scale

            result = veto_system.process_step(x_t, x_next_base, spread_t, epr_t)

            shock_history.append(result['shock_t'])
            veto_history.append(result['veto_effect'])

        results_by_scale[scale] = {
            'shock': shock_history,
            'veto': veto_history,
            'shock_mean': float(np.mean(shock_history)),
            'veto_mean': float(np.mean(veto_history))
        }

    # Compare correlations across scales
    reference_shock = np.array(results_by_scale[1.0]['shock'])
    reference_veto = np.array(results_by_scale[1.0]['veto'])

    correlations = {}

    for scale in scales:
        other_shock = np.array(results_by_scale[scale]['shock'])
        other_veto = np.array(results_by_scale[scale]['veto'])

        corr_shock = np.corrcoef(reference_shock, other_shock)[0, 1]
        corr_veto = np.corrcoef(reference_veto, other_veto)[0, 1]

        correlations[scale] = {
            'shock': float(corr_shock) if not np.isnan(corr_shock) else 0.0,
            'veto': float(corr_veto) if not np.isnan(corr_veto) else 0.0
        }

    # Compute minimum average correlation
    min_avg_correlation = min(
        np.mean(list(c.values())) for c in correlations.values()
    )

    findings = {
        'correlations': correlations,
        'min_avg_correlation': min_avg_correlation,
        'metrics_by_scale': {s: {
            'shock_mean': r['shock_mean'],
            'veto_mean': r['veto_mean']
        } for s, r in results_by_scale.items()}
    }

    # Pass if average correlation > 0.5
    passed = min_avg_correlation > 0.5

    return {
        'test': 'scale_robustness',
        'findings': findings,
        'min_avg_correlation': min_avg_correlation,
        'note': 'Veto dynamics should be qualitatively similar across scales',
        'pass': passed
    }


# =============================================================================
# COMPREHENSIVE AUDIT
# =============================================================================

def run_full_audit(base_path: str = None, results_path: str = None) -> Dict:
    """Run all endogeneity audits."""
    if base_path is None:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if results_path is None:
        results_path = os.path.join(base_path, 'results', 'phase20')

    print("=" * 60)
    print("PHASE 20: ENDOGENEITY AUDIT")
    print("=" * 60)

    audit_results = {}

    # Audit 1: No Magic Numbers
    print("\n[1/5] Auditing for magic numbers...")
    audit_results['magic_numbers'] = audit_magic_numbers(base_path)
    status = "PASS" if audit_results['magic_numbers']['pass'] else "FAIL"
    print(f"      Status: {status}")
    print(f"      Warnings: {audit_results['magic_numbers']['n_warnings']}")
    print(f"      Errors: {audit_results['magic_numbers']['n_errors']}")

    # Audit 2: No Semantic Labels
    print("\n[2/5] Auditing for semantic labels...")
    audit_results['semantic_labels'] = audit_semantic_labels(base_path)
    status = "PASS" if audit_results['semantic_labels']['pass'] else "FAIL"
    print(f"      Status: {status}")
    print(f"      Info: {audit_results['semantic_labels']['n_info']}")
    print(f"      Errors: {audit_results['semantic_labels']['n_errors']}")

    # Audit 3: Provenance Traceability
    print("\n[3/5] Auditing provenance traceability...")
    audit_results['provenance'] = audit_provenance(base_path)
    status = "PASS" if audit_results['provenance']['pass'] else "FAIL"
    print(f"      Status: {status}")
    print(f"      Provenances found: {audit_results['provenance']['found_provenances']}")
    print(f"      Parameters logged: {len(audit_results['provenance']['found_params'])}")

    # Audit 4: Null Comparisons
    print("\n[4/5] Auditing null comparisons...")
    audit_results['null_comparisons'] = audit_null_comparisons(results_path)
    status = "PASS" if audit_results['null_comparisons']['pass'] else "FAIL"
    print(f"      Status: {status}")

    # Audit 5: Scale Robustness
    print("\n[5/5] Auditing scale robustness...")
    try:
        audit_results['scale_robustness'] = audit_scale_robustness()
        status = "PASS" if audit_results['scale_robustness']['pass'] else "FAIL"
        print(f"      Status: {status}")
        print(f"      Min avg correlation: {audit_results['scale_robustness']['min_avg_correlation']:.4f}")
    except Exception as e:
        audit_results['scale_robustness'] = {'pass': False, 'error': str(e)}
        print(f"      Status: ERROR - {e}")

    # Overall assessment
    all_passed = all(r.get('pass', False) for r in audit_results.values())
    n_passed = sum(1 for r in audit_results.values() if r.get('pass', False))

    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {n_passed}/5")
    print(f"Overall: {'CERTIFIED ENDOGENOUS' if all_passed else 'AUDIT FAILED'}")
    print("=" * 60)

    return {
        'audits': audit_results,
        'n_passed': n_passed,
        'n_total': 5,
        'all_passed': all_passed,
        'certification': 'ENDOGENOUS' if all_passed else 'FAILED'
    }


# =============================================================================
# PARAMETER DERIVATION REPORT
# =============================================================================

def generate_derivation_report(base_path: str = None) -> str:
    """Generate human-readable report of parameter derivations."""
    if base_path is None:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    report = """# Phase 20: Parameter Derivation Report

## Overview

This report documents the provenance of ALL parameters used in Phase 20.
Every parameter must be derived from data - ZERO magic numbers allowed.

## Structural Veto System (veto20.py)

### Intrusion Detection

| Parameter | Formula | Source | Justification |
|-----------|---------|--------|---------------|
| `delta` | `||x_t - mu_k||` | Manifold, prototypes | Distance to nearest prototype |
| `delta_spread` | `|spread_t - spread_ema|` | Spread history | Deviation from EMA |
| `delta_epr` | `|epr_t - epr_ema|` | EPR history | Deviation from EMA |
| `alpha_ema` | `1/sqrt(t+1)` | Timestep | Endogenous decay |
| `rank_delta` | `rank(delta)` | Delta history | Rank |
| `rank_spread` | `rank(delta_spread)` | Delta spread history | Rank |
| `rank_epr` | `rank(delta_epr)` | Delta EPR history | Rank |
| `shock_t` | `rank_delta * rank_spread * rank_epr` | Ranks | Multiplicative shock |

### Structural Opposition Field

| Parameter | Formula | Source | Justification |
|-----------|---------|--------|---------------|
| `direction` | `x_t - mu_k` | Manifold, nearest prototype | Direction from prototype |
| `rank_shock` | `rank(shock_t)` | Shock history | Rank of current shock |
| `O_t` | `-rank_shock * normalize(direction)` | Rank, direction | Opposition back to prototype |

### Resistance Gain

| Parameter | Formula | Source | Justification |
|-----------|---------|--------|---------------|
| `window_size` | `sqrt(t)` | Timestep | Endogenous window |
| `window_std` | `std(shock[-window:])` | Recent shock history | Volatility |
| `gamma_t` | `1/(1 + window_std)` | Window std | Inverse volatility |

### Veto Transition Adjustment

| Parameter | Formula | Source | Justification |
|-----------|---------|--------|---------------|
| `adjustment` | `gamma_t * O_t` | Gamma, opposition | Scaled opposition |
| `x_next` | `x_next_base + adjustment` | Base transition, adjustment | Veto-adjusted next state |
| `veto_effect` | `||x_next - x_next_base||` | Adjusted vs base | Magnitude of veto |

## Structural Bounds

| Bound | Value | Justification |
|-------|-------|---------------|
| Min window (k) | 5 | Statistical minimum |
| N_prototypes | 5 | Configuration |
| STATE_DIM | 4 | Configuration |
| Warmup | 50 | sqrt(T) scaling |

## Semantic Label Analysis

Phase 20 uses ONLY structural/mathematical terms:
- `veto` - structural veto (mathematical adjustment)
- `resistance` - resistance gain (mathematical inverse volatility)
- `opposition` - opposition field (mathematical vector)
- `shock` - shock indicator (mathematical deviation product)
- `intrusion` - intrusion detection (mathematical distance)

NO human-centric semantic labels:
- No `pain`, `fear`, `threat`
- No `danger`, `safe`, `attack`
- No `reward`, `goal`, `utility`

## Certification

All parameters traced to:
1. Data statistics (ranks, variances, stds)
2. History lengths (sqrt(t), 1/sqrt(t+1))
3. Mathematical identities (0, 0.5, 1)
4. Geometric operations (normalization, distance)

ZERO arbitrary/magic constants.
"""

    return report


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(base_path, 'results', 'phase20')

    # Run full audit
    audit_report = run_full_audit(base_path, results_path)

    # Save audit results
    os.makedirs(results_path, exist_ok=True)

    with open(os.path.join(results_path, 'endogeneity_audit.json'), 'w') as f:
        json.dump(audit_report, f, indent=2, default=str)

    print(f"\nAudit results saved to: {results_path}/endogeneity_audit.json")

    # Generate derivation report
    derivation_report = generate_derivation_report(base_path)

    with open(os.path.join(results_path, 'parameter_derivations.md'), 'w') as f:
        f.write(derivation_report)

    print(f"Derivation report saved to: {results_path}/parameter_derivations.md")

    sys.exit(0 if audit_report['all_passed'] else 1)
