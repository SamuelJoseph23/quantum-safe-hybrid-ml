"""
Differential Privacy Implementation using Laplace Mechanism
Adds privacy guarantees to federated learning gradients
"""

import numpy as np
from typing import Dict, Tuple, Any
import json


class DifferentialPrivacy:
    """
    Implements Differential Privacy using Laplace mechanism.
    
    Adds calibrated noise to gradients to achieve (ε, δ)-differential privacy.
    Based on: Dwork et al., "Differential Privacy: A Survey of Results"
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5,
                 sensitivity: float = 1.0, noise_type: str = 'gaussian'):
        """
        Initialize DP mechanism.
        
        Args:
            epsilon: Privacy budget (smaller = more private, but noisier)
                     Common values: 0.5 (strong), 1.0 (moderate), 8.0 (weak)
            delta: Probability of privacy violation (1e-5 is standard)
            sensitivity: L2 sensitivity of gradient (bound on max change)
            noise_type: 'laplace' or 'gaussian'
        
        Privacy Interpretation:
            - epsilon = 1.0, delta = 1e-5: Strong privacy (recommended)
            - epsilon = 8.0, delta = 1e-5: Moderate privacy
            - epsilon > 10: Weak privacy (minimal protection)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.noise_type = noise_type
        
        # Calculate scale parameter for noise
        self.scale = self._calculate_scale()
        
        # Privacy accounting
        self.privacy_spent = 0.0  # Track cumulative privacy budget
        self.rounds_executed = 0
        
        print(f"✓ Differential Privacy initialized")
        print(f"  - Privacy budget (ε): {self.epsilon}")
        print(f"  - Failure probability (δ): {self.delta}")
        print(f"  - Noise scale: {self.scale:.6f}")
        print(f"  - Mechanism: {self.noise_type.capitalize()}")
    
    def _calculate_scale(self) -> float:
        """
        Calculate noise scale parameter based on mechanism.
        
        For Laplace: scale = sensitivity / epsilon
        For Gaussian: scale = sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon
        """
        if self.noise_type == 'laplace':
            # Laplace mechanism: simpler, faster, standard for FL
            return self.sensitivity / self.epsilon
        
        elif self.noise_type == 'gaussian':
            # Gaussian mechanism: better for large epsilon
            factor = np.sqrt(2 * np.log(1.25 / self.delta))
            return factor * self.sensitivity / self.epsilon
        
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
    
    def set_sensitivity(self, sensitivity: float) -> None:
        """
        Update sensitivity (typically tied to clipping norm) and recompute scale.
        """
        self.sensitivity = float(sensitivity)
        self.scale = self._calculate_scale()

    def account_step(self, epsilon_spent: float | None = None) -> None:
        """
        Account for one privacy mechanism application.
        """
        self.privacy_spent += float(self.epsilon if epsilon_spent is None else epsilon_spent)
        self.rounds_executed += 1

    def add_noise(self, gradient: np.ndarray, *, account: bool = False) -> np.ndarray:
        """
        Add differential privacy noise to gradient.
        
        Args:
            gradient: Model gradient (numpy array)
        
        Returns:
            Noisy gradient with same shape
        """
        # Generate noise
        if self.noise_type == 'laplace':
            # Laplace distribution: Lap(0, scale)
            noise = np.random.laplace(0, self.scale, size=gradient.shape)
        else:
            # Gaussian distribution: N(0, scale^2)
            noise = np.random.normal(0, self.scale, size=gradient.shape)
        
        noisy_gradient = gradient + noise
        if account:
            self.account_step()
        return noisy_gradient
    
    def add_noise_to_dict(self, gradient_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Add noise to a dictionary of gradients (for multi-layer models).
        
        Args:
            gradient_dict: Dict of {param_name: gradient_array}
        
        Returns:
            Dict of {param_name: noisy_gradient_array}
        """
        noisy_dict = {}
        for param_name, gradient in gradient_dict.items():
            noisy_dict[param_name] = self.add_noise(gradient)
        
        return noisy_dict
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """
        Return privacy accounting information.
        
        Returns:
            Dict with privacy metrics
        """
        return {
            "epsilon_per_round": self.epsilon,
            "total_privacy_spent": self.privacy_spent,
            "rounds_executed": self.rounds_executed,
            "delta": self.delta,
            "noise_mechanism": self.noise_type,
            "noise_scale": self.scale,
            "privacy_level": self._interpret_epsilon(self.epsilon)
        }
    
    @staticmethod
    def _interpret_epsilon(epsilon: float) -> str:
        """
        Interpret epsilon value for human readability.
        
        Args:
            epsilon: Privacy budget
        
        Returns:
            Description of privacy level
        """
        if epsilon < 0.1:
            return "Extremely Strong (ε < 0.1)"
        elif epsilon < 0.5:
            return "Very Strong (0.1 ≤ ε < 0.5)"
        elif epsilon < 1.0:
            return "Strong (0.5 ≤ ε < 1.0) [RECOMMENDED]"
        elif epsilon < 5.0:
            return "Moderate (1.0 ≤ ε < 5.0)"
        elif epsilon < 10.0:
            return "Weak (5.0 ≤ ε < 10.0)"
        else:
            return "Very Weak (ε ≥ 10.0)"


class DPAnalyzer:
    """
    Analyzes privacy-utility tradeoffs.
    Tracks how DP noise affects model accuracy.
    """
    
    def __init__(self):
        """Initialize analyzer."""
        self.metrics = {
            'rounds': [],
            'epsilon_values': [],
            'accuracy': [],
            'privacy_spent': [],
            'noise_magnitude': []
        }
    
    def record_round(self, round_num: int, epsilon: float, accuracy: float,
                     privacy_spent: float, noise_mag: float):
        """
        Record metrics for a training round.
        
        Args:
            round_num: Round number
            epsilon: Epsilon for this round
            accuracy: Model accuracy achieved
            privacy_spent: Cumulative privacy budget spent
            noise_mag: Average magnitude of noise added
        """
        self.metrics['rounds'].append(round_num)
        self.metrics['epsilon_values'].append(epsilon)
        self.metrics['accuracy'].append(accuracy)
        self.metrics['privacy_spent'].append(privacy_spent)
        self.metrics['noise_magnitude'].append(noise_mag)
    
    def get_tradeoff_analysis(self) -> Dict[str, Any]:
        """
        Analyze privacy-utility tradeoff.
        
        Returns:
            Dict with tradeoff metrics
        """
        if len(self.metrics['accuracy']) < 2:
            return {"status": "Not enough data"}
        
        accuracy_drop = self.metrics['accuracy'][0] - self.metrics['accuracy'][-1]
        final_privacy = self.metrics['privacy_spent'][-1]
        
        return {
            "initial_accuracy": self.metrics['accuracy'][0],
            "final_accuracy": self.metrics['accuracy'][-1],
            "accuracy_drop": accuracy_drop,
            "accuracy_drop_percentage": (accuracy_drop / self.metrics['accuracy'][0]) * 100,
            "total_privacy_budget_spent": final_privacy,
            "rounds_executed": len(self.metrics['rounds']),
            "avg_noise_magnitude": np.mean(self.metrics['noise_magnitude']),
            "max_noise_magnitude": np.max(self.metrics['noise_magnitude'])
        }
    
    def save_report(self, filename: str = '../results/metrics/dp_analysis.json'):
        """Save analysis to JSON file."""
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        report = {
            "metrics": self.metrics,
            "tradeoff_analysis": self.get_tradeoff_analysis()
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=float)
        
        print(f"✓ DP analysis saved to {filename}")


class DPGradientClipper:
    """
    Implements gradient clipping for DP.
    Ensures gradients have bounded L2 norm (sensitivity bound).
    """
    
    def __init__(self, max_norm: float = 1.0):
        """
        Initialize gradient clipper.
        
        Args:
            max_norm: Maximum L2 norm of gradient
        """
        self.max_norm = max_norm
    
    def clip(self, gradient: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Clip gradient to max norm.
        
        Args:
            gradient: Model gradient
        
        Returns:
            (Clipped gradient, clipping factor used)
        """
        norm = np.linalg.norm(gradient)
        
        if norm > self.max_norm:
            clipping_factor = self.max_norm / norm
            clipped = gradient * clipping_factor
            return clipped, clipping_factor
        else:
            return gradient, 1.0
    
    def clip_dict(self, gradient_dict: Dict[str, np.ndarray]) -> \
                  Tuple[Dict[str, np.ndarray], float]:
        """
        Clip all gradients in a dictionary.
        
        Args:
            gradient_dict: Dict of gradients
        
        Returns:
            (Dict of clipped gradients, max clipping factor used)
        """
        clipped_dict = {}
        max_factor = 1.0
        
        for param_name, gradient in gradient_dict.items():
            clipped, factor = self.clip(gradient)
            clipped_dict[param_name] = clipped
            max_factor = max(max_factor, factor)
        
        return clipped_dict, max_factor
