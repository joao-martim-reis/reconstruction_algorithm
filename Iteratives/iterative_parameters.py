"""
ITERATIVE ALGORITHM PARAMETERS

This file contains configurations for iterative algorithms available in TIGRE.
To use a specific algorithm in the MAIN script, pick its name or a preset and
the parameters will be loaded automatically.

"""


ALGORITHM_CONFIGS = {

    # BASIC ALGORITHMS (no TV regularization)

    'SIRT': {
        'category': 'basic',
        'function_name': 'sirt',
        'iterations': 100,
        'description': 'Classic simultaneous iterative technique. Good balance of quality and speed.',
        'params': {}
    },
    
    'CGLS': {
        'category': 'basic',
        'function_name': 'cgls',
        'iterations': 50,
        'description': 'Conjugate gradient least squares. Fast convergence, good for fine details.',
        'params': {}
    },
    
    'LSQR': {
        'category': 'basic',
        'function_name': 'lsqr',
        'iterations': 50,
        'description': 'Least squares with QR. Numerically stable.',
        'params': {}
    },
    
    'LSMR': {
        'category': 'basic',
        'function_name': 'lsmr',
        'iterations': 50,
        'description': 'Improved version of LSQR. Faster and stable.',
        'params': {}
    },
    
    'OSSART': {
        'category': 'basic',
        'function_name': 'ossart',
        'iterations': 30,
        'description': 'SART with ordered subsets. Very fast for previews.',
        'params': {
            'blocksize': 20  # increase = faster | decrease = better quality
        }
    },
    
    'SART': {
        'category': 'basic',
        'function_name': 'sart',
        'iterations': 80,
        'description': 'Algebraic variant similar to SIRT with different weighting.',
        'params': {}
    },
    
    'MLEM': {
        'category': 'basic',
        'function_name': 'mlem',
        'iterations': 30,
        'description': 'Maximum Likelihood Expectation Maximization. Statistical method for low-dose data.',
        'params': {}
    },

    'OSSART_TV': {
        'category': 'tv',
        'function_name': 'ossart_tv',
        'iterations': 50,
        'description': 'OSSART with TV regularization. Fast and reduces metal artifacts.',
        'params': {
            'blocksize': 5,
            #'tvlambda': 10, # TV weight: increase = stronger smoothing | decrease = more detail
            #'tviter': 25    # internal TV iterations: increase = stronger smoothing | decrease = more detail
        }
    },
    
    'SART_TV': {
        'category': 'tv',
        'function_name': 'sart_tv',
        'iterations': 60,
        'description': 'SART with TV regularization. Slower than OSSART_TV but more precise.',
        'params': {

        }
    },
    
    'ASD_POCS': {
        'category': 'tv',
        'function_name': 'asd_pocs',
        'iterations': 120,
        'description': 'Adaptive steepest descent with TV. Good for severe artifacts.',
        'params': {
            'alpha': 0.002    # ASD alpha (descent step): increase = faster convergence
        }
    },
    
    'AWASD_POCS': {
        'category': 'tv',
        'function_name': 'awasd_pocs',
        'iterations': 100,
        'description': 'Adaptive weighted ASD_POCS with automatic weight adjustment.',
        'params': {
            'alpha': 0.002
        }
    },
}



def get_algorithm_config(algorithm_name):
    """
    Return the full configuration for an algorithm.
    """
    if algorithm_name not in ALGORITHM_CONFIGS:
        available = list(ALGORITHM_CONFIGS.keys())
        raise ValueError(
            f"Algorithm '{algorithm_name}' not found!\n"
            f"Available algorithms: {available}"
        )

    config = ALGORITHM_CONFIGS[algorithm_name].copy()
    config['algorithm_name'] = algorithm_name
    return config



