# advanced_math.py
"""
Custom Advanced Mathematical Simulation based on SEDPNR Model

Govindankutty, Sreeraag & P G, Shynu. (2024). Epidemic modeling for misinformation spread in digital networks through a social intelligence approach. 
Scientific Reports. 14. 10.1038/s41598-024-69657-0. 

Fractional Calculus
Bayesian Belief Updates
Game Theory Utility Functions
Cognitive Science elements for Human Decision Making
"""
import numpy as np
from scipy.special import gamma as gamma_func
from typing import Tuple, List, Optional
import config as cfg
def fractional_probability(base_rate: float, time_in_state: int, v: float = None) -> float:
    """
    Implements fractional dynamics with memory effects.
    
    The probability of transitioning decays following a power law,
    simulating how humans "get used to" stimuli over time.
    
    Based on the Mittag-Leffler function which appears in fractional
    differential equations modeling memory-dependent systems.
    
    Args:
        base_rate: Base transition probability (0-1)
        time_in_state: How long the agent has been in current state
        v: Fractional order (0 < v ≤ 1), lower = more memory
    
    Returns:
        Adjusted probability (0-1)
    
    Mathematical basis:
        For v < 1, probability decay follows: P(t) ∝ t^(v-1)
        This gives "heavy tail" behavior - slow decay after initial period
    """
    v = v or cfg.FRACTIONAL_ORDER_V
    
    if time_in_state <= 0:
        return base_rate
    
    # Power law decay simulating Mittag-Leffler function tail
    # For v=1: no memory effect (exponential decay)
    # For v<1: heavy tail (algebraic decay)
    memory_factor = (time_in_state) ** (v - 1)
    
    # Normalize and bound
    prob = base_rate * memory_factor
    return np.clip(prob, 0.001, 0.999)


def mittag_leffler_approx(alpha: float, t: float, terms: int = 10) -> float:
    """
    Approximate the Mittag-Leffler function E_α(t).
    
    The Mittag-Leffler function generalizes the exponential and
    appears naturally in fractional calculus solutions.
    
    E_α(t) = Σ_{k=0}^∞ t^k / Γ(αk + 1)
    
    Args:
        alpha: Fractional order (0 < α ≤ 1)
        t: Time or argument
        terms: Number of terms in series expansion
    
    Returns:
        Approximate value of E_α(t)
    """
    result = 0.0
    for k in range(terms):
        denominator = gamma_func(alpha * k + 1)
        if denominator > 0:
            result += (t ** k) / denominator
    return result


def caputo_derivative(f_values: List[float], dt: float = 1.0, alpha: float = None) -> float:
    """
    Approximate the Caputo fractional derivative at the last point.
    
    The Caputo derivative is commonly used in modeling because it
    allows traditional initial conditions and respects causality.
    
    Args:
        f_values: History of function values
        dt: Time step size
        alpha: Fractional order
    
    Returns:
        Approximate Caputo derivative
    """
    alpha = alpha or cfg.FRACTIONAL_ORDER_V
    n = len(f_values)
    
    if n < 2:
        return 0.0
    
    # Simple finite difference approximation
    weights = []
    for j in range(n - 1):
        # Weight based on distance from current time
        weight = ((n - 1 - j) ** (1 - alpha) - (n - 2 - j) ** (1 - alpha)) if j < n-1 else 1
        weights.append(weight)
    
    derivative = 0.0
    for j in range(n - 1):
        derivative += weights[j] * (f_values[j + 1] - f_values[j])
    
    coefficient = dt ** (-alpha) / gamma_func(2 - alpha)
    return coefficient * derivative

# Bayesian Inference
def bayesian_belief_update(
    prior: float,
    signal_is_share: bool,
    rep_sender: float,
    noise: float = 0.0
) -> float:
    """
    Bayesian belief update for consumer agents.
    
    Uses Bayes' Theorem to update belief about content truthfulness
    given a signal (share) and the reputation of the sender.
    
    P(True | Share) = P(Share | True) * P(True) / P(Share)
    
    Args:
        prior: Prior probability that content is true (0-1)
        signal_is_share: Whether we observed a share action
        rep_sender: Reputation of the sender (0-1)
        noise: Random noise to add (simulates bounded rationality)
    
    Returns:
        Posterior probability (0-1)
    """
    if not signal_is_share:
        return prior  # No signal, no update
    
    # Map sender reputation to conditional probabilities
    # High reputation → more likely sharing truth
    # Low reputation → more likely sharing lies
    
    # P(Share | True) - truthful people share true things
    p_share_given_truth = 0.1 + (0.8 * rep_sender)
    
    # P(Share | False) - liars share false things
    p_share_given_lie = 0.9 - (0.8 * rep_sender)
    
    # Bayes' theorem
    numerator = p_share_given_truth * prior
    denominator = (p_share_given_truth * prior) + (p_share_given_lie * (1.0 - prior))
    
    if denominator < 1e-10:
        return prior
    
    posterior = numerator / denominator
    
    # Add noise for bounded rationality
    if noise > 0:
        posterior += np.random.uniform(-noise, noise)
    
    return np.clip(posterior, 0.0, 1.0)


def confirmation_bias_update(
    prior: float,
    new_evidence: float,
    bias_strength: float
) -> float:
    """
    Apply confirmation bias to belief updating.
    
    Humans tend to weight confirming evidence more heavily than
    disconfirming evidence. This function models that asymmetry.
    
    Args:
        prior: Current belief (0-1)
        new_evidence: Strength of new evidence (0=against, 1=for)
        bias_strength: How biased the agent is (0=rational, 1=very biased)
    
    Returns:
        Updated belief with confirmation bias applied
    """
    # Standard Bayesian would move toward evidence
    rational_update = 0.5 * prior + 0.5 * new_evidence
    
    # Confirming evidence (same direction as prior)
    if (prior > 0.5 and new_evidence > 0.5) or (prior < 0.5 and new_evidence < 0.5):
        # Weight confirming evidence more heavily
        weight = 0.5 + 0.3 * bias_strength
        biased_update = weight * new_evidence + (1 - weight) * prior
    else:
        # Disconfirming evidence - weight it less
        weight = 0.5 - 0.3 * bias_strength
        biased_update = weight * new_evidence + (1 - weight) * prior
    
    return np.clip(biased_update, 0.0, 1.0)

# Game Theory Adversarial Layer

def platform_utility(
    engagement: float,
    moderation_level: float,
    misinformation_count: int
) -> float:
    """
    Calculate Platform's utility function (Stackelberg leader).
    
    U_P = φ₁·E - φ₂·C(M) - φ₃·D
    
    Where:
    - E = engagement level
    - C(M) = cost of moderation (convex)
    - D = damage from misinformation
    
    Args:
        engagement: Current user engagement level
        moderation_level: Platform's moderation intensity (0-1)
        misinformation_count: Number of users spreading misinfo
    
    Returns:
        Utility value (higher is better for platform)
    """
    # Convex moderation cost: harder to catch everything
    moderation_cost = moderation_level ** cfg.MOD_COST_CONVEXITY
    
    # Linear engagement benefit
    engagement_benefit = cfg.PHI_ENGAGEMENT * engagement
    
    # Moderation cost
    mod_cost = cfg.PHI_MOD_COST * moderation_cost
    
    # Reputational damage from misinformation
    rep_damage = cfg.PHI_REP_DAMAGE * misinformation_count
    
    return engagement_benefit - mod_cost - rep_damage


def calculate_disinformant_utility(
    spread: float,
    moderation_level: float,
    reputation: float
) -> float:
    """
    Calculate Disinformant's utility function (Stackelberg follower).
    
    U_D = Spread - α · Risk
    
    Where Risk = f(Moderation, Reputation)
    
    Args:
        spread: Expected spread of misinformation
        moderation_level: Platform's moderation level (0-1)
        reputation: Disinformant's current reputation (0-1)
    
    Returns:
        Utility value (higher = more incentive to share)
    """
    # Risk increases with moderation
    # Risk decreases with reputation (harder to catch trusted sources)
    risk_factor = (moderation_level * cfg.ALPHA_PENALTY) / (reputation + 0.1)
    
    # Utility is spread minus expected penalty
    utility = spread - risk_factor
    
    return utility


def nash_bargaining_solution(
    utility_player1: List[float],
    utility_player2: List[float],
    disagreement_point: Tuple[float, float] = (0, 0)
) -> int:
    """
    Find the Nash Bargaining Solution from available outcomes.
    
    The NBS maximizes the product of utility gains over the
    disagreement point, representing "fair" division of surplus.
    
    Args:
        utility_player1: Player 1's utility for each outcome
        utility_player2: Player 2's utility for each outcome
        disagreement_point: Fallback utilities if no agreement
    
    Returns:
        Index of the Nash Bargaining Solution outcome
    """
    best_idx = 0
    best_product = -np.inf
    
    for i in range(len(utility_player1)):
        u1 = utility_player1[i] - disagreement_point[0]
        u2 = utility_player2[i] - disagreement_point[1]
        
        if u1 > 0 and u2 > 0:
            product = u1 * u2
            if product > best_product:
                best_product = product
                best_idx = i
    
    return best_idx

# Network Analysis

def calculate_spectral_radius(adjacency_matrix: np.ndarray) -> float:
    """
    Calculate the spectral radius (largest eigenvalue) of the network.
    
    The spectral radius determines the epidemic threshold:
    - If R0 > 1/spectral_radius, epidemic spreads
    - If R0 < 1/spectral_radius, epidemic dies out
    
    Args:
        adjacency_matrix: Network adjacency matrix
    
    Returns:
        Spectral radius
    """
    eigenvalues = np.linalg.eigvalsh(adjacency_matrix)
    return max(abs(eigenvalues))

def calculate_algebraic_connectivity(laplacian: np.ndarray) -> float:
    """
    Calculate algebraic connectivity (Fiedler value) of the network.
    
    This is the second-smallest eigenvalue of the Laplacian.
    Higher values = better connected network = faster spreading.
    
    Args:
        laplacian: Graph Laplacian matrix
    
    Returns:
        Algebraic connectivity
    """
    eigenvalues = sorted(np.linalg.eigvalsh(laplacian))
    return eigenvalues[1] if len(eigenvalues) > 1 else 0.0
    
# Cognitive Science Modules

def prospect_theory_weight(probability: float, gamma: float = 0.7) -> float:
    """
    Apply Prospect Theory probability weighting.
    
    Humans overweight small probabilities and underweight large ones.
    This function applies the Prelec (1998) weighting function.
    
    w(p) = exp(-(-ln(p))^γ)
    
    Args:
        probability: Objective probability (0-1)
        gamma: Curvature parameter (0.5-1.0, lower = more distortion)
    
    Returns:
        Subjective decision weight
    """
    if probability <= 0:
        return 0.0
    if probability >= 1:
        return 1.0
    
    return np.exp(-((-np.log(probability)) ** gamma))


def social_proof_effect(
    own_belief: float,
    neighbor_beliefs: List[float],
    conformity: float
) -> float:
    """
    Model social proof / conformity effects on belief.
    
    People adjust their beliefs toward the group mean,
    weighted by their conformity tendency.
    
    Args:
        own_belief: Agent's current belief
        neighbor_beliefs: List of neighbor beliefs
        conformity: Agent's conformity tendency (0-1)
    
    Returns:
        Adjusted belief after social influence
    """
    if not neighbor_beliefs:
        return own_belief
    
    group_mean = np.mean(neighbor_beliefs)
    # Weighted average between own belief and group
    adjusted = (1 - conformity) * own_belief + conformity * group_mean
    return np.clip(adjusted, 0.0, 1.0)
    
def emotional_amplification(
    base_probability: float,
    emotional_intensity: float,
    emotion_type: str = "fear"
) -> float:
    """
    Model how emotions amplify or dampen decision probabilities.
    Fear accelerates action, calm slows it.
    Args:
        base_probability: Base action probability
        emotional_intensity: Intensity of emotion (0-1)
        emotion_type: Type of emotion (fear, anger, calm, etc.)
    
    Returns:
        Emotionally-modified probability
    """
    multipliers = {
        "fear": 1.0 + 0.5 * emotional_intensity,      # Fear speeds action
        "anger": 1.0 + 0.3 * emotional_intensity,     # Anger speeds action
        "calm": 1.0 - 0.2 * emotional_intensity,      # Calm slows action
        "skeptical": 1.0 - 0.4 * emotional_intensity, # Skepticism inhibits
        "curious": 1.0 + 0.2 * emotional_intensity,   # Curiosity motivates
    }
    
    multiplier = multipliers.get(emotion_type, 1.0)
    return np.clip(base_probability * multiplier, 0.0, 1.0)
def decision_fatigue(
    base_quality: float,
    decisions_made: int,
    max_capacity: int = 10
) -> float:
    """
    Model decision fatigue - quality decreases with more decisions.
    After many decisions, agents make poorer choices.
    Args:
        base_quality: Base decision quality (0-1)
        decisions_made: Number of decisions already made
        max_capacity: Mental capacity before significant degradation
    Returns:
        Degraded decision quality
    """
    fatigue_factor = max(0.5, 1.0 - (decisions_made / max_capacity) * 0.5)
    return base_quality * fatigue_factor
