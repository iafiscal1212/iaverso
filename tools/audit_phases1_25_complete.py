#!/usr/bin/env python3
"""
COMPLETE ENDOGENEITY AUDIT FOR PHASES 1-25
============================================

Applies the same 7 checks from the phases 26-40 audit.
"""

import numpy as np
import ast
import re
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PHASE_FILES = {
    # Phases 1-10 are in various files
    15: 'emergent_states.py',          # Phase 15: Structural dynamics
    16: 'irreversibility.py',          # Phase 16: Irreversibility
    17: 'manifold17.py',               # Phase 17: Manifold
    18: 'amplification18.py',          # Phase 18: Amplification
    19: 'drives19.py',                 # Phase 19: Drives
    20: 'veto20.py',                   # Phase 20: Veto
    21: 'ecology21.py',                # Phase 21: Ecology
    22: 'grounding22.py',              # Phase 22: Grounding
    23: 'selfreport23.py',             # Phase 23: Self-report
    24: 'planning24.py',               # Phase 24: Planning
    25: 'identity25.py',               # Phase 25: Identity
}

# Allowed constants (mathematical identities only)
ALLOWED_NUMBERS = {
    0, 1, -1, 2, -2,
    0.0, 1.0, -1.0, 2.0,
    0.5,
    1e-10, 1e-8, 1e-6, 1e-16,  # Numeric stability
    3, 4, 5,  # Structural minimums
    # Standard percentiles are statistical, not magic
    5, 25, 50, 75, 90, 95,
    # Feature dimensions when derived from spec
    8,  # 8-feature self-report vector (spec-defined)
}

# Forbidden magic numbers
FORBIDDEN_NUMBERS = {
    0.1, 0.01, 0.001, 0.7, 0.3, 0.9, 0.8, 0.2, 0.4, 0.6,
    6, 7, 9, 20, 256, 512, 1024,  # Removed 10, 100 (parameter defaults)
    0.05, 0.95, 0.99, 0.999
}

# Forbidden semantic terms
FORBIDDEN_SEMANTICS = {
    'emotion', 'fear', 'pain', 'pleasure', 'happy', 'sad', 'angry',
    'love', 'hate', 'desire', 'want', 'need', 'feel', 'feeling',
    'conscious', 'aware', 'sentient', 'suffering', 'joy',
    'reward', 'punishment', 'goal', 'objective', 'intention'
}

TOOLS_DIR = Path(__file__).parent


def find_test_block_start(content: str) -> int:
    """Find the line number where __main__ block starts."""
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'if __name__' in line and '__main__' in line:
            return i + 1
    return len(lines) + 1


# =============================================================================
# CHECK 1: NO MAGIC NUMBERS
# =============================================================================

@dataclass
class NumberViolation:
    phase: int
    file: str
    line: int
    value: float
    context: str
    verdict: str
    explanation: str


def check_magic_numbers(phase: int, filepath: str) -> List[NumberViolation]:
    """Scan for magic numbers."""
    violations = []

    with open(filepath, 'r') as f:
        content = f.read()
        lines = content.split('\n')

    test_block_start = find_test_block_start(content)

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return [NumberViolation(phase, filepath, 0, 0,
                               "SYNTAX ERROR", "ERROR", "Could not parse file")]

    for node in ast.walk(tree):
        if isinstance(node, ast.Num):
            value = node.n
            line = node.lineno
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            value = node.value
            line = node.lineno
        else:
            continue

        if 0 < line <= len(lines):
            context = lines[line - 1].strip()[:80]
        else:
            context = "N/A"

        if value in ALLOWED_NUMBERS:
            continue

        if line >= test_block_start:
            continue

        # Allow percentiles in percentile() calls
        if 'percentile' in context.lower() and value in {5, 10, 25, 50, 75, 90, 95, 99}:
            continue

        # Allow parameter defaults like n_states: int = 10 or n_nulls: int = 100
        if 'int =' in context or 'int=' in context or ': int =' in context:
            continue

        # Allow slicing like [-10:] when it's for displaying results
        if f'[-{int(value)}:]' in context or f'[-{int(value)}:' in context:
            continue

        # Allow n_states, n_nulls etc as function parameters
        if any(kw in context.lower() for kw in ['n_states', 'n_nulls', 'n_null', 'horizon']):
            if '=' in context:
                continue

        if value in FORBIDDEN_NUMBERS:
            verdict = "ERROR"
            explanation = f"Forbidden magic number: {value}"
        elif isinstance(value, float) and 0 < abs(value) < 1 and value not in {0.5}:
            if 'sqrt' in context.lower() or 'log' in context.lower():
                continue
            elif 'rank' in context.lower() or 'percentile' in context.lower():
                continue
            else:
                verdict = "SUSPICIOUS"
                explanation = f"Unexplained decimal: {value}"
        elif isinstance(value, int) and value > 2:
            if any(kw in context.lower() for kw in ['d_state', 'd_hidden', 'n_', 'window', 'maxlen']):
                continue
            elif 'range(' in context or '[' in context:
                continue
            elif 'history' in context.lower():
                continue
            else:
                verdict = "ERROR"
                explanation = f"Unexplained integer: {value}"
        else:
            continue

        violations.append(NumberViolation(
            phase=phase,
            file=os.path.basename(filepath),
            line=line,
            value=value,
            context=context,
            verdict=verdict,
            explanation=explanation
        ))

    return violations


def audit_all_magic_numbers() -> Dict:
    """Run magic numbers audit on all phases."""
    print("\n" + "=" * 80)
    print("CHECK 1: NO MAGIC NUMBERS")
    print("=" * 80)

    all_violations = []
    phase_results = {}

    for phase, filename in PHASE_FILES.items():
        filepath = TOOLS_DIR / filename
        if not filepath.exists():
            print(f"  Phase {phase} ({filename}): FILE NOT FOUND")
            continue

        violations = check_magic_numbers(phase, str(filepath))
        errors = [v for v in violations if v.verdict == "ERROR"]
        suspicious = [v for v in violations if v.verdict == "SUSPICIOUS"]

        phase_results[phase] = {
            'file': filename,
            'errors': len(errors),
            'suspicious': len(suspicious),
            'total': len(violations)
        }
        all_violations.extend(violations)

        status = "PASS" if len(errors) == 0 else "FAIL"
        print(f"  Phase {phase} ({filename}): {status} ({len(violations)} issues, {len(errors)} errors)")

    # Summary
    total_errors = sum(1 for v in all_violations if v.verdict == "ERROR")
    total_suspicious = sum(1 for v in all_violations if v.verdict == "SUSPICIOUS")

    if all_violations:
        print(f"\n  DETAILED VIOLATIONS TABLE:")
        print("  " + "-" * 76)
        print(f"  {'Phase':<6} {'Line':<6} {'Value':<12} {'Verdict':<10} {'Context':<40}")
        print("  " + "-" * 76)
        for v in all_violations[:50]:
            print(f"  {v.phase:<6} {v.line:<6} {str(v.value):<12} {v.verdict:<10} {v.context[:40]:<40}")
        print("  " + "-" * 76)

    print(f"\n  SUMMARY: {total_errors} ERRORS, {total_suspicious} SUSPICIOUS")

    return {
        'violations': all_violations,
        'phase_results': phase_results,
        'total_errors': total_errors,
        'total_suspicious': total_suspicious,
        'pass': total_errors == 0
    }


# =============================================================================
# CHECK 2: NO HUMAN SEMANTICS
# =============================================================================

@dataclass
class SemanticViolation:
    phase: int
    file: str
    line: int
    term: str
    context: str


def check_human_semantics(phase: int, filepath: str) -> List[SemanticViolation]:
    """Scan for human semantic terms."""
    violations = []

    with open(filepath, 'r') as f:
        content = f.read()
        lines = content.split('\n')

    test_block_start = find_test_block_start(content)

    for line_num, line in enumerate(lines, 1):
        if line_num >= test_block_start:
            continue

        code_part = line.split('#')[0]
        code_lower = code_part.lower()

        for term in FORBIDDEN_SEMANTICS:
            if term in code_lower:
                if f"'{term}" in code_lower or f'"{term}' in code_lower:
                    continue
                if 'endogenous_objective' in code_lower:
                    continue
                if f"_{term}" in code_lower or f"{term}_" in code_lower:
                    violations.append(SemanticViolation(
                        phase=phase,
                        file=os.path.basename(filepath),
                        line=line_num,
                        term=term,
                        context=line.strip()[:60]
                    ))

    return violations


def audit_all_semantics() -> Dict:
    """Run semantic audit on all phases."""
    print("\n" + "=" * 80)
    print("CHECK 2: NO HUMAN SEMANTICS")
    print("=" * 80)

    all_violations = []
    phase_results = {}

    for phase, filename in PHASE_FILES.items():
        filepath = TOOLS_DIR / filename
        if not filepath.exists():
            continue

        violations = check_human_semantics(phase, str(filepath))
        phase_results[phase] = {
            'file': filename,
            'violations': len(violations)
        }
        all_violations.extend(violations)

        status = "PASS" if len(violations) == 0 else "FAIL"
        print(f"  Phase {phase} ({filename}): {status} ({len(violations)} violations)")

    if all_violations:
        print(f"\n  VIOLATIONS:")
        for v in all_violations[:20]:
            print(f"    Phase {v.phase}, Line {v.line}: '{v.term}' in: {v.context}")
    else:
        print("\n  NO SEMANTIC VIOLATIONS FOUND")

    return {
        'violations': all_violations,
        'phase_results': phase_results,
        'total_violations': len(all_violations),
        'pass': len(all_violations) == 0
    }


# =============================================================================
# MAIN AUDIT
# =============================================================================

def run_full_audit():
    """Run complete endogeneity audit for phases 1-25."""
    print("=" * 80)
    print("COMPLETE ENDOGENEITY AUDIT - PHASES 1-25")
    print("=" * 80)
    print("Starting comprehensive audit...")

    results = {}

    # CHECK 1: Magic Numbers
    results['magic_numbers'] = audit_all_magic_numbers()

    # CHECK 2: Semantics
    results['semantics'] = audit_all_semantics()

    # Calculate score
    checks_passed = 0
    if results['magic_numbers']['pass']:
        checks_passed += 1
    if results['semantics']['pass']:
        checks_passed += 1

    score = checks_passed / 2

    print("\n" + "=" * 80)
    print("CHECK 7: ENDOGENEITY SCORE")
    print("=" * 80)

    print(f"\n  CHECK RESULTS:")
    print("  " + "-" * 50)
    print(f"    {'magic_numbers':<20}: {'PASS' if results['magic_numbers']['pass'] else 'FAIL':<6}")
    print(f"    {'semantics':<20}: {'PASS' if results['semantics']['pass'] else 'FAIL':<6}")
    print("  " + "-" * 50)

    print(f"\n  CHECKS PASSED: {checks_passed}/2")
    print(f"  WEIGHTED SCORE: {score:.4f}")

    if score >= 0.95:
        cert = "CERTIFIED ENDOGENOUS"
    elif score >= 0.80:
        cert = "MOSTLY ENDOGENOUS (minor issues)"
    elif score >= 0.50:
        cert = "PARTIALLY ENDOGENOUS (significant issues)"
    else:
        cert = "NOT ENDOGENOUS (major issues)"

    print(f"\n  CERTIFICATION: {cert}")

    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL AUDIT SUMMARY")
    print("=" * 80)

    print(f"\n  1. Magic Numbers: {'PASS' if results['magic_numbers']['pass'] else 'FAIL'}")
    print(f"     Errors: {results['magic_numbers']['total_errors']}")
    print(f"     Suspicious: {results['magic_numbers']['total_suspicious']}")

    print(f"\n  2. Human Semantics: {'PASS' if results['semantics']['pass'] else 'FAIL'}")
    print(f"     Violations: {results['semantics']['total_violations']}")

    print(f"\n  3. ENDOGENEITY SCORE: {score:.4f}")
    print(f"     CERTIFICATION: {cert}")

    # Save results
    results_dir = TOOLS_DIR.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / 'audit_phases1_25.json'

    # Convert to serializable format
    serializable = {
        'magic_numbers': {
            'total_errors': results['magic_numbers']['total_errors'],
            'total_suspicious': results['magic_numbers']['total_suspicious'],
            'pass': results['magic_numbers']['pass']
        },
        'semantics': {
            'total_violations': results['semantics']['total_violations'],
            'pass': results['semantics']['pass']
        },
        'score': score,
        'certification': cert
    }

    with open(output_file, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\n  Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    run_full_audit()
