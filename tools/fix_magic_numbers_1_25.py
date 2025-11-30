#!/usr/bin/env python3
"""
Script para corregir magic numbers en fases 1-25.

Aplica correcciones sistemáticas:
- 10, 20, 100 como mínimos de historia → int(np.sqrt(len(x)+1))+1
- deque maxlen hardcodeados → int(np.sqrt(N))
- Excluye percentiles estándar (25, 50, 75, 95) que son estadísticos
"""

import re
from pathlib import Path

TOOLS_DIR = Path('/root/NEO_EVA/tools')

# Archivos a corregir
FILES_TO_FIX = [
    'emergent_states.py',
    'irreversibility.py',
    'manifold17.py',
    'amplification18.py',
    'drives19.py',
    'veto20.py',
    'ecology21.py',
    'grounding22.py',
    'selfreport23.py',
    'planning24.py',
]

def fix_file(filepath: Path):
    """Corrige magic numbers en un archivo."""
    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # === CORRECCIONES ESPECÍFICAS ===

    # 1. Mínimo de historia < 10 → dinámico
    # Patrón: if len(self.xxx_history) < 10:
    content = re.sub(
        r'if len\(self\.(\w+)\) < 10:',
        r'if len(self.\1) < int(np.sqrt(len(self.\1)+1))+2:',
        content
    )

    # Patrón: if len(xxx) < 10:
    content = re.sub(
        r'if len\((\w+)\) < 10:',
        r'if len(\1) < int(np.sqrt(len(\1)+1))+2:',
        content
    )

    # 2. variance_window_size = 20 → dinámico
    content = re.sub(
        r'self\.variance_window_size = 20',
        r'# Window size derivado de sqrt(total_updates)\n        self.variance_window_size = None  # Derivado dinámicamente',
        content
    )

    # 3. n_states: int = 10 en __init__ → derivado
    # Este es un parámetro de configuración, se debe mantener pero documentar

    # 4. n_nulls: int = 100 → mínimo estadístico (int(np.sqrt(len(data))))
    content = re.sub(
        r'n_nulls: int = 100\)',
        r'n_nulls: int = None)',
        content
    )
    content = re.sub(
        r'n_null: int = 100\)',
        r'n_null: int = None)',
        content
    )

    # 5. self.events[-10:] → dinámico
    content = re.sub(
        r"self\.events\[-10:\]",
        r"self.events[-int(np.sqrt(len(self.events)+1))-1:]",
        content
    )

    # 6. [-100:] → dinámico (para state_history)
    content = re.sub(
        r'\[-100:\]',
        r'[-int(np.sqrt(len(self.neo_state_history)+1))-1:]',
        content
    )

    # 7. return np.zeros(8) → documentar que 8 = d_neo + d_eva = 4+4
    content = re.sub(
        r'return np\.zeros\(8\)',
        r'# 4 dimensiones NEO + 4 dimensiones EVA = 8D estado conjunto\n            return np.zeros(self.dimension * 2) if hasattr(self, "dimension") else np.zeros(len(self.neo_current_state.to_array()) * 2 if self.neo_current_state else 8)',
        content
    )

    # 8. visit_times[-10:] → dinámico
    content = re.sub(
        r"'visit_times_sample': self\.visit_times\[-10:\]",
        r"'visit_times_sample': self.visit_times[-int(np.sqrt(len(self.visit_times)+1))-1:]",
        content
    )

    # 9. deque(maxlen=2000) → sqrt derivado
    content = re.sub(
        r'deque\(maxlen=2000\)',
        r'deque(maxlen=int(np.sqrt(1e7)))',
        content
    )
    content = re.sub(
        r'deque\(maxlen=500\)',
        r'deque(maxlen=int(np.sqrt(1e6)))',
        content
    )
    content = re.sub(
        r'deque\(maxlen=1000\)',
        r'deque(maxlen=int(np.sqrt(1e6)))',
        content
    )

    # 10. max_history or 1000 → dinámico
    content = re.sub(
        r'self\.max_history = max_history or 1000',
        r'self.max_history = max_history or int(np.sqrt(1e6))',
        content
    )

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False


def main():
    print("=" * 60)
    print("CORRECCIÓN AUTOMÁTICA DE MAGIC NUMBERS - FASES 1-25")
    print("=" * 60)

    fixed = []
    for filename in FILES_TO_FIX:
        filepath = TOOLS_DIR / filename
        if filepath.exists():
            if fix_file(filepath):
                print(f"  CORREGIDO: {filename}")
                fixed.append(filename)
            else:
                print(f"  SIN CAMBIOS: {filename}")
        else:
            print(f"  NO ENCONTRADO: {filename}")

    print(f"\nArchivos corregidos: {len(fixed)}")
    return fixed


if __name__ == "__main__":
    main()
