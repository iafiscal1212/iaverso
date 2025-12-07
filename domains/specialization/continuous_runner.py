"""
EJECUCIÓN CONTINUA 24/7
=======================

RUN 24/7
DO NOT TOUCH CORE
LOG EVERYTHING
PASS TESTS EVERY SESSION
NO INTERPRETATION
NO TABLES UNTIL ASKED
"""

import subprocess
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ContinuousRunner:
    """
    Runner de ejecución continua.

    NO modifica el núcleo.
    NO interpreta resultados.
    SOLO ejecuta, observa y registra.
    """

    def __init__(self, base_dir: str = "logs/observation/sessions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.session_count = 0

    def run_session(self, rounds: int = 30, agents: List[str] = None) -> Dict[str, Any]:
        """
        Ejecuta una sesión completa.

        1. Reset session context
        2. Run N rounds
        3. Log
        4. Run endogeneity tests
        5. Archive
        """
        from domains.specialization.tera_nucleus import TeraDirector
        from domains.specialization.endogenous_observer import EndogenousObserver

        if agents is None:
            agents = ['AGENT_001', 'AGENT_002', 'AGENT_003']

        # Session ID
        session_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%SZ")
        session_dir = self.base_dir / f"session_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        self.session_count += 1

        # Create director (CORE - NO MODIFICATION)
        director = TeraDirector(seed=None)  # Random seed each session
        director.start_session(agents)

        # Create observer (READ-ONLY)
        observer = EndogenousObserver(log_dir=session_dir)
        observer.start_session(session_id)

        # Run rounds
        for round_num in range(1, rounds + 1):
            results = director.run_round()
            observer.observe_round(round_num, results)

        # Run endogeneity tests
        test_results = self._run_tests()

        # Save observation log
        observer.save_log()

        # Save test results
        self._save_test_results(session_dir, test_results)

        # Save metadata
        metadata = {
            'session_id': session_id,
            'session_number': self.session_count,
            'rounds': rounds,
            'tasks_total': rounds * len(agents),
            'agents': agents,
            'observer_mode': 'READ_ONLY',
            'core_modified': False,
            'tests_executed': True,
            'tests_passed': test_results['passed'],
            'tests_failed': test_results['failed'],
            'timestamp_start': observer.get_session().start_time if observer.get_session() else session_id,
            'timestamp_end': datetime.now().isoformat(),
            'notes': 'none',
        }
        self._save_metadata(session_dir, metadata)

        return {
            'session_id': session_id,
            'session_dir': str(session_dir),
            'rounds': rounds,
            'tests': test_results,
        }

    def _run_tests(self) -> Dict[str, Any]:
        """Ejecuta tests de endogeneidad."""
        test_files = [
            'tests/test_tera_hard_fail.py',
            'tests/test_endogenous_hard_fail.py',
            'tests/test_tension_hard_rules.py',
        ]

        results = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'files': [],
        }

        for test_file in test_files:
            try:
                proc = subprocess.run(
                    ['python3', '-m', 'pytest', test_file, '-v', '--tb=no', '-q'],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(Path(__file__).parent.parent.parent),
                )

                file_result = {
                    'file': test_file,
                    'returncode': proc.returncode,
                    'passed': 0,
                    'failed': 0,
                }

                # Parse output
                for line in proc.stdout.split('\n'):
                    if 'passed' in line:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if 'passed' in p and i > 0:
                                try:
                                    count = int(parts[i-1])
                                    file_result['passed'] = count
                                    results['passed'] += count
                                    results['total_tests'] += count
                                except:
                                    pass
                    if 'failed' in line:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if 'failed' in p and i > 0:
                                try:
                                    count = int(parts[i-1])
                                    file_result['failed'] = count
                                    results['failed'] += count
                                    results['total_tests'] += count
                                except:
                                    pass

                results['files'].append(file_result)

            except Exception as e:
                results['files'].append({
                    'file': test_file,
                    'error': str(e),
                })

        return results

    def _save_test_results(self, session_dir: Path, results: Dict[str, Any]):
        """Guarda resultados de tests."""
        path = session_dir / "endogeneity_tests.yaml"
        with open(path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False, allow_unicode=True)

    def _save_metadata(self, session_dir: Path, metadata: Dict[str, Any]):
        """Guarda metadata de sesión."""
        path = session_dir / "session_metadata.yaml"
        with open(path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)

    def run_continuous(self, rounds_per_session: int = 30, max_sessions: int = None):
        """
        Ejecuta sesiones continuamente.

        NO pausa.
        NO solicita confirmación.
        NO resume hasta que se le pida.
        """
        session_num = 0

        while max_sessions is None or session_num < max_sessions:
            session_num += 1

            result = self.run_session(rounds=rounds_per_session)

            # Minimal output (no interpretation)
            print(f"SESSION {session_num}: {result['session_id']} | tests: {result['tests']['passed']}/{result['tests']['total_tests']}")


def run_single_session(rounds: int = 30) -> Dict[str, Any]:
    """Ejecuta una sesión única."""
    runner = ContinuousRunner()
    return runner.run_session(rounds=rounds)


def run_continuous(rounds_per_session: int = 30, max_sessions: int = None):
    """Inicia ejecución continua."""
    runner = ContinuousRunner()
    runner.run_continuous(rounds_per_session=rounds_per_session, max_sessions=max_sessions)


if __name__ == "__main__":
    # Default: run continuous
    run_continuous(rounds_per_session=30)
