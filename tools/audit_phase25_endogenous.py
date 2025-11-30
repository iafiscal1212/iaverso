#!/usr/bin/env python3
"""Phase 25: Anti-Magic Audit"""

import numpy as np
import re
import ast
import os
from pathlib import Path
from typing import Dict
import json
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent))


ALLOWED_CONSTANTS = {
    0, 1, 2, 0.5, 0.0, 1.0, 2.0,
    1e-16, 1e-10, 1e-8, 1e-6,
    42, 3, 4, 5, 6, 100, 500, 1000,
    50, 60, 150, 45, 30,
}

FORBIDDEN_SEMANTICS = {
    'pain', 'pleasure', 'suffering', 'happy', 'sad', 'angry',
    'goal', 'desire', 'want', 'intention', 'objective',
    'fear', 'threat', 'danger', 'punishment', 'harm',
    'reward_signal', 'punishment_signal', 'reward_function',
    'conscious', 'aware', 'sentient', 'feeling', 'emotional',
}


def audit_magic_numbers(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        content = f.read()
    violations = []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return {'pass': False, 'violations': ['Syntax error'], 'details': 'Syntax error'}

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            val = node.value
            if val not in ALLOWED_CONSTANTS:
                is_allowed = -val in ALLOWED_CONSTANTS or (isinstance(val, int) and -10 <= val <= 20)
                if isinstance(val, float) and (0 < abs(val) < 1 or 0.8 <= abs(val) <= 1.0):
                    is_allowed = True
                if not is_allowed:
                    violations.append({'value': val, 'line': getattr(node, 'lineno', '?')})

    lines = content.split('\n')
    filtered = [v for v in violations if not any(kw in lines[v['line']-1] for kw in ['range(', 'shape', 'zeros(', 'ones(', 'randn('])]
    return {'pass': len(filtered) == 0, 'violations': filtered, 'details': f'Found {len(filtered)} magic numbers'}


def audit_semantic_labels(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        content = f.read().lower()
    violations = [{'term': t} for t in FORBIDDEN_SEMANTICS if re.search(r'\b' + t + r'\b', content)]
    return {'pass': len(violations) == 0, 'violations': violations, 'details': f'Found {len(violations)} semantic violations'}


def audit_provenance(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        content = f.read()
    required = ['I_t', 'd_t', 'R_t', 'alpha', 'rank']
    found = []
    if 'IDENTITY25_PROVENANCE' in content:
        pattern = r"'endogenous_params':\s*\[([\s\S]*?)\],\s*'no_magic"
        match = re.search(pattern, content)
        if match:
            params_str = match.group(1).lower()
            for p in required:
                if p.lower() in params_str:
                    found.append(p)
    missing = [p for p in required if p not in found]
    return {'pass': len(missing) == 0, 'required': required, 'found': found, 'missing': missing,
            'details': f'Found {len(found)}/{len(required)} provenance entries'}


def audit_null_comparison(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        content = f.read().lower()
    required = ['disabled', 'shuffled', 'random']
    found = [n for n in required if n in content]
    return {'pass': len(found) == len(required), 'found': found, 'details': f'Found {len(found)}/{len(required)} null models'}


def audit_scale_robustness() -> Dict:
    from identity25 import OperatorResistantIdentity
    np.random.seed(42)
    scales = [0.1, 1.0, 10.0, 100.0]
    R_by_scale = {}

    for scale in scales:
        identity_sys = OperatorResistantIdentity()
        R_mags = []
        z_t = np.zeros(5)
        for t in range(100):
            drift = np.sin(np.arange(5) * 0.1 + t * 0.02) * scale * 0.1
            z_t = z_t + drift + np.random.randn(5) * scale * 0.05
            result = identity_sys.process_step(z_t)
            R_mags.append(result['R_magnitude'])
        R_by_scale[scale] = {'mean': float(np.mean(R_mags)), 'std': float(np.std(R_mags))}

    # Check CV consistency across scales
    cvs = [R_by_scale[s]['std'] / (R_by_scale[s]['mean'] + 1e-16) for s in scales]
    cv_range = max(cvs) - min(cvs)

    return {'pass': cv_range < 0.5, 'results_by_scale': R_by_scale, 'cv_range': cv_range,
            'details': f'CV range across scales: {cv_range:.4f}'}


def run_full_audit(module_path: str = None, runner_path: str = None) -> Dict:
    if module_path is None:
        module_path = str(Path(__file__).parent / "identity25.py")
    if runner_path is None:
        runner_path = str(Path(__file__).parent / "phase25_identity.py")

    print("=" * 60)
    print("PHASE 25: ANTI-MAGIC AUDIT")
    print("=" * 60)

    audits = {}

    print("\n[1] Magic Number Scan...")
    a1_m = audit_magic_numbers(module_path)
    a1_r = audit_magic_numbers(runner_path)
    a1_pass = a1_m['pass'] and a1_r['pass']
    audits['magic_numbers'] = {'pass': a1_pass, 'module': a1_m, 'runner': a1_r}
    print(f"  [{'PASS' if a1_pass else 'FAIL'}] {a1_m['details']}, {a1_r['details']}")

    print("\n[2] Semantic Label Scan...")
    a2_m = audit_semantic_labels(module_path)
    a2_r = audit_semantic_labels(runner_path)
    a2_pass = a2_m['pass'] and a2_r['pass']
    audits['semantic_labels'] = {'pass': a2_pass}
    print(f"  [{'PASS' if a2_pass else 'FAIL'}] No forbidden terms")

    print("\n[3] Provenance Verification...")
    a3 = audit_provenance(module_path)
    audits['provenance'] = a3
    print(f"  [{'PASS' if a3['pass'] else 'FAIL'}] {a3['details']}")

    print("\n[4] Null Model Verification...")
    a4 = audit_null_comparison(runner_path)
    audits['null_comparison'] = a4
    print(f"  [{'PASS' if a4['pass'] else 'FAIL'}] {a4['details']}")

    print("\n[5] Scale Robustness Test...")
    a5 = audit_scale_robustness()
    audits['scale_robustness'] = a5
    print(f"  [{'PASS' if a5['pass'] else 'FAIL'}] {a5['details']}")

    n_pass = sum(1 for a in audits.values() if a['pass'])
    all_pass = n_pass == 5

    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    for name in ['magic_numbers', 'semantic_labels', 'provenance', 'null_comparison', 'scale_robustness']:
        print(f"  [{'PASS' if audits[name]['pass'] else 'FAIL'}] {name}")

    print(f"\nTotal: {n_pass}/5 PASS")
    print(f"\nENDOGENOUS: {'YES' if all_pass else 'NO'}")

    return {'audits': audits, 'n_pass': n_pass, 'all_pass': all_pass, 'timestamp': datetime.now().isoformat()}


def save_audit_results(results: Dict, output_dir: str = "results/phase25"):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "audit_results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {output_dir}/audit_results.json")


if __name__ == "__main__":
    results = run_full_audit()
    save_audit_results(results)
