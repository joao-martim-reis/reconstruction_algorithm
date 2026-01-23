"""
ITERATIVE ALGORITHM PARAMETERS

This file contains configurations for all iterative algorithms available in TIGRE.
To use a specific algorithm in the MAIN script, pick its name or a preset and
the parameters will be loaded automatically.

HOW TO USE:
-----------
In `MAIN_TIGRE_iterative.py`, choose an algorithm via `get_algorithm_config()` or
select a preset via `get_preset_config()`. The algorithm configuration will be
returned as a dictionary containing sensible defaults and explanation strings.

    # PARAMETER GUIDE (brief):
    #- `iterations`: number of algorithm iterations (typical 30-200). Increase to
    #  improve convergence or reduce to speed up.
    #- `blocksize`: number of projections per subset for OSSART (typical 10-40).
    #- `lmbda`: TV regularization weight (typical 5-50). Higher = stronger smoothing.
    #- `ng`: internal TV gradient iterations (typical 15-30).
    #- `alpha`: descent step for ASD_POCS (typical 0.001-0.005).
    #- `epsilon`: fidelity tolerance for ASD_POCS (typical 0.01-0.1).

RECOMMENDATIONS:
- Simple phantoms (PMMA, water): `SIRT` or `CGLS`.
- Resolution tests (bar phantom): `CGLS` or `OSSART`.
- Metal artifacts: `OSSART_TV` or `SART_TV`.
- Very noisy data: `SART_TV` with higher `tv_lambda`.
- Fast preview: `OSSART` with large `blocksize`.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#                      ALGORITMOS BÁSICOS (SEM TV)
# ═══════════════════════════════════════════════════════════════════════════

ALGORITHM_CONFIGS = {
    
    # ─────────────────────────────────────────────────────────────────────────
    # SIRT - Simultaneous Iterative Reconstruction Technique
    # ─────────────────────────────────────────────────────────────────────────
    'SIRT': {
        'category': 'basic',
        'iterations': 100,
        'description': 'Classic simultaneous iterative technique. Good balance of quality and speed.',
        'best_for': 'PMMA phantoms, water phantoms, general cases without severe artifacts',
        'params': {}
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # CGLS - Conjugate Gradient Least Squares
    # ─────────────────────────────────────────────────────────────────────────
    'CGLS': {
        'category': 'basic',
        'iterations': 50,
        'description': 'Conjugate gradient least squares. Fast convergence, good for fine details.',
        'best_for': 'Bar pattern, simple phantoms, when speed is desired',
        'params': {}
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # LSQR - Least Squares QR
    # ─────────────────────────────────────────────────────────────────────────
    'LSQR': {
        'category': 'basic',
        'iterations': 50,
        'description': 'Least squares with QR. Numerically stable.',
        'best_for': 'Ill-conditioned problems, noisy data',
        'params': {}
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # LSMR - Least Squares Minimal Residual
    # ─────────────────────────────────────────────────────────────────────────
    'LSMR': {
        'category': 'basic',
        'iterations': 50,
        'description': 'Improved version of LSQR. Faster and stable.',
        'best_for': 'Alternative to LSQR for large problems',
        'params': {}
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # OSSART - Ordered Subsets SART
    # ─────────────────────────────────────────────────────────────────────────
    'OSSART': {
        'category': 'basic',
        'iterations': 30,
        'description': 'SART with ordered subsets. Very fast for previews.',
        'best_for': 'Quick reconstructions, when you have many projections',
        'params': {
            'blocksize': 20  # increase = faster | decrease = better quality
        }
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # SART - Simultaneous Algebraic Reconstruction Technique
    # ─────────────────────────────────────────────────────────────────────────
    'SART': {
        'category': 'basic',
        'iterations': 50,
        'description': 'Algebraic variant similar to SIRT with different weighting.',
        'best_for': 'Alternative to SIRT',
        'params': {}
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#                   ALGORITHMS WITH TOTAL VARIATION (TV)
# ═══════════════════════════════════════════════════════════════════════════

ALGORITHM_CONFIGS.update({
    
    # ─────────────────────────────────────────────────────────────────────────
    # OSSART_TV - OSSART com Total Variation
    # ─────────────────────────────────────────────────────────────────────────
    'OSSART_TV': {
        'category': 'tv',
        'iterations': 100,
        'description': 'OSSART with TV regularization. Fast and reduces metal artifacts.',
        'best_for': 'Recommended for metal artifacts and fast TV-regularized reconstructions',
        'params': {
            'blocksize': 20,      # increase = faster | decrease = better
            'lmbda': 20.0,    # TV weight: increase = stronger smoothing | decrease = more detail
            'ng': 25           # internal TV iterations (usually 20-25)
        }
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # SART_TV - SART com Total Variation
    # ─────────────────────────────────────────────────────────────────────────
    'SART_TV': {
        'category': 'tv',
        'iterations': 60,
        'description': 'SART with TV regularization. Slower than OSSART_TV but more precise.',
        'best_for': 'Noisy data when processing time is available',
        'params': {
            'lmbda': 15.0,    # TV weight: increase = stronger smoothing | decrease = more detail
            'ng': 25           # internal TV iterations
        }
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # ASD_POCS - Adaptive Steepest Descent - Projection Onto Convex Sets
    # ─────────────────────────────────────────────────────────────────────────
    'ASD_POCS': {
        'category': 'tv',
        'iterations': 40,
        'description': 'Adaptive steepest descent with TV. Good for severe artifacts.',
        'best_for': 'Severe artifacts, incomplete data',
        'params': {
            'alpha': 0.002,   # ASD alpha (descent step): increase = faster convergence
            'epsilon': 0.05,  # ASD epsilon (fidelity tolerance): increase = more smoothing
            'ng': 25           # internal TV iterations
        }
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # AWASD_POCS - Adaptive Weighted ASD_POCS
    # ─────────────────────────────────────────────────────────────────────────
    'AWASD_POCS': {
        'category': 'tv',
        'iterations': 40,
        'description': 'Adaptive weighted ASD_POCS with automatic weight adjustment.',
        'best_for': 'Alternative to ASD_POCS with automatic adaptation',
        'params': {
            'alpha': 0.002,
            'epsilon': 0.05,
            'ng': 25
        }
    },
})


# ═══════════════════════════════════════════════════════════════════════════
#                         CONFIGURAÇÕES PRÉ-DEFINIDAS
# ═══════════════════════════════════════════════════════════════════════════

# Configurações otimizadas para casos comuns
PRESET_CONFIGS = {
    
    'fantoma_pmma': {
        'algorithm': 'SIRT',
        'modifications': {'iterations': 80},
        'description': 'Optimized for simple PMMA phantoms'
    },
    
    'fantoma_agua': {
        'algorithm': 'CGLS',
        'modifications': {'iterations': 60},
        'description': 'Optimized for water phantoms'
    },
    
    'padrao_barras': {
        'algorithm': 'CGLS',
        'modifications': {'iterations': 50},
        'description': 'Optimized for resolution tests (bar pattern)'
    },
    
    'metal_artifacts': {
        'algorithm': 'OSSART_TV',
        'modifications': {
            'iterations': 50,
            'params': {'tv_lambda': 25.0, 'blocksize': 20, 'tv_ng': 25}
        },
        'description': 'Optimized to remove metal artifacts'
    },
    
    'ruido_alto': {
        'algorithm': 'SART_TV',
        'modifications': {
            'iterations': 70,
            'params': {'tv_lambda': 30.0, 'tv_ng': 25}
        },
        'description': 'Optimized for high-noise data'
    },
    
    'preview_rapido': {
        'algorithm': 'OSSART',
        'modifications': {
            'iterations': 20,
            'params': {'blocksize': 40}
        },
        'description': 'Quick reconstruction for preview'
    },
    
    'maxima_qualidade': {
        'algorithm': 'SIRT',
        'modifications': {'iterations': 200},
        'description': 'Maximum quality (slow)'
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#                              FUNÇÕES ÚTEIS
# ═══════════════════════════════════════════════════════════════════════════

def get_algorithm_config(algorithm_name):
    """
    Return the full configuration for an algorithm.

    Args:
        algorithm_name: Algorithm name (e.g., 'SIRT', 'OSSART_TV')

    Returns:
        dict: Algorithm configuration
    """
    if algorithm_name not in ALGORITHM_CONFIGS:
        available = list(ALGORITHM_CONFIGS.keys())
        raise ValueError(
            f"Algorithm '{algorithm_name}' not found!\n"
            f"Available algorithms: {available}"
        )

    return ALGORITHM_CONFIGS[algorithm_name].copy()


def get_preset_config(preset_name):
    """
    Return a preset configuration for common cases.

    Args:
        preset_name: Preset name (e.g., 'fantoma_pmma', 'metal_artifacts')

    Returns:
        dict: Full algorithm configuration with preset modifications applied
    """
    if preset_name not in PRESET_CONFIGS:
        available = list(PRESET_CONFIGS.keys())
        raise ValueError(
            f"Preset '{preset_name}' not found!\n"
            f"Available presets: {available}"
        )

    preset = PRESET_CONFIGS[preset_name]
    config = get_algorithm_config(preset['algorithm'])

    # Apply preset modifications
    if 'modifications' in preset:
        for key, value in preset['modifications'].items():
            if key == 'params':
                config['params'].update(value)
            else:
                config[key] = value

    return config


def list_available_algorithms():
    """Print all available algorithms with descriptions."""
    print("\n" + "="*80)
    print("AVAILABLE ALGORITHMS".center(80))
    print("="*80)
    
    print("\n📊 BASIC ALGORITHMS (no TV regularization):")
    print("-" * 80)
    for name, cfg in ALGORITHM_CONFIGS.items():
        if cfg['category'] == 'basic':
            print(f"\n  🔹 {name}")
            print(f"     Default iterations: {cfg['iterations']}")
            print(f"     Description: {cfg['description']}")
            print(f"     Best for: {cfg['best_for']}")
    
    print("\n\n🎯 TV ALGORITHMS (Total Variation - reduces artifacts):")
    print("-" * 80)
    for name, cfg in ALGORITHM_CONFIGS.items():
        if cfg['category'] == 'tv':
            print(f"\n  🔹 {name}")
            print(f"     Default iterations: {cfg['iterations']}")
            print(f"     Description: {cfg['description']}")
            print(f"     Best for: {cfg['best_for']}")
    
    print("\n" + "="*80 + "\n")


def list_presets():
    """List all available presets."""
    print("\n" + "="*80)
    print("PRESET CONFIGURATIONS".center(80))
    print("="*80)
    
    for name, preset in PRESET_CONFIGS.items():
        print(f"\n  🎯 {name}")
        print(f"     Algorithm: {preset['algorithm']}")
        print(f"     Description: {preset['description']}")
    
    print("\n" + "="*80 + "\n")


def print_algorithm_info(algorithm_name):
    """Print detailed information about a specific algorithm."""
    config = get_algorithm_config(algorithm_name)
    
    print("\n" + "="*80)
    print(f"INFO: {algorithm_name}".center(80))
    print("="*80)
    
    print(f"\nCategory: {'🎯 TV-Regularized' if config['category'] == 'tv' else '📊 Basic'}")
    print(f"Default iterations: {config['iterations']}")
    print(f"Description: {config['description']}")
    print(f"Best for: {config['best_for']}")
    
    if config['params']:
        print(f"\nConfigurable parameters:")
        for param, value in config['params'].items():
            print(f"  • {param}: {value}")
    else:
        print(f"\nNo additional configurable parameters.")
    
    print("\n" + "="*80 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#                              TESTES
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(__doc__)
    
    # List all algorithms
    list_available_algorithms()
    
    # List presets
    list_presets()
    
    # Usage example
    print("\n" + "="*80)
    print("USAGE EXAMPLE".center(80))
    print("="*80)
    
    print("\n# Get configuration for an algorithm:")
    print("config = get_algorithm_config('OSSART_TV')")
    config = get_algorithm_config('OSSART_TV')
    print(f"Result: {config}")
    
    print("\n\n# Get preset for metal artifacts:")
    print("config = get_preset_config('metal_artifacts')")
    config = get_preset_config('metal_artifacts')
    print(f"Result: {config}")
    
    print("\n\n# Print detailed info for an algorithm:")
    print("print_algorithm_info('SIRT')")
    print_algorithm_info('SIRT')
