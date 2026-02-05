# config.py
"""
Central configuration for the Neuro-Symbolic Defense System.

This file contains all tunable parameters organized by category:
- Simulation settings
- Network topology
- Fractional dynamics (memory effects)
- Personality → Behavior mappings
- Game theory payoffs
- Defense system thresholds
"""

import numpy as np

DATA_FILENAME = 'survey_data.csv'
OUTPUT_DIR = '.'
GRID_WIDTH = 50
GRID_HEIGHT = 50
MAX_STEPS = 200
SEED = 42
PRINT_EVERY = 20
SAVE_CHECKPOINTS = False
CHECKPOINT_EVERY = 50
FRACTIONAL_ORDER_V = 0.8
MEMORY_DECAY_RATE = 0.95  # How fast old experiences fade
EMOTIONAL_PERSISTENCE = 0.9  # How long emotions linger
BETA_INTERCEPT = 0.35
BETA_WEIGHTS = {
    'openness': 0.15,           # Open people explore new ideas (including bad ones)
    'conscientiousness': -0.12,  # Conscientious people think before acting
    'neuroticism': 0.10,         # Neurotic people may share out of anxiety
    'extraversion': 0.08,        # Extraverts have more connections
    'agreeableness': 0.03,       # Agreeable people go along with crowd
}
LAMBDA_INTERCEPT = 0.2
LAMBDA_WEIGHTS = {
    'openness': 0.05,            # Curious people process faster
    'conscientiousness': 0.20,   # Conscientious = thorough processing
    'neuroticism': -0.08,        # Anxiety slows deliberation
    'extraversion': 0.03,
    'agreeableness': 0.0,
}
SHARE_WEIGHTS = {
    'extraversion': 0.25,        # Extraverts share more
    'conscientiousness': -0.20,  # Conscientious people filter more
    'neuroticism': 0.10,         # Anxiety can drive sharing
    'openness': 0.05,
    'agreeableness': 0.05,
}
GAMMA_BASE = 0.15
EPSILON_BASE = 0.25
REINFECTION_RESISTANCE = 0.7
PRIOR_TRUTH_PROB = 0.5
THRESHOLD_LOW = 0.35   # Below this = believe the lie (become spreader)
THRESHOLD_HIGH = 0.65  # Above this = reject the lie (recover)
FEAR_THRESHOLD_SHIFT = 0.10     # Fear lowers the threshold
SKEPTICISM_THRESHOLD_SHIFT = 0.15  # Skepticism raises it
ALPHA_PENALTY = 2.5
REP_DELAY_FACTOR = 0.9
DISINFO_REP_GAIN_SUCCESS = 0.05    # Rep gain per successful spread
DISINFO_REP_LOSS_CAUGHT = 0.20     # Rep loss when debunked
DISINFO_REP_RECOVERY = 0.01        # Rep recovery during hold
PHI_ENGAGEMENT = 1.2      # Weight on user engagement
PHI_MOD_COST = 0.8        # Weight on moderation cost
PHI_REP_DAMAGE = 2.0      # Weight on reputational damage from misinfo
MOD_COST_CONVEXITY = 1.5
TRUST_LOSS_PER_MISINFO = 0.02    # Trust lost per % of infected users
TRUST_GAIN_PER_CLEAN = 0.01     # Trust gained when clean
FACTCHECK_ENERGY_COST = 0.15      # Energy per debunk
FACTCHECK_ENERGY_RECOVERY = 0.10  # Energy recovered per idle step
FACTCHECK_REP_GAIN_ACTIVE = 0.03   # Rep gain per debunk (up to limit)
FACTCHECK_REP_LOSS_INACTIVE = 0.02  # Rep loss during crisis inaction
FACTCHECK_SPAM_THRESHOLD = 5       # Max debunks before seeming spammy
# Barabási-Albert scale-free network parameters
NETWORK_AVG_DEGREE = 4    # Average connections per node (m parameter)
IDENTIFY_HUBS = True      # Find and track hub nodes
HUB_PERCENTILE = 95       # Top X% by degree = hubs
VSCORE_THRESHOLD_MONITOR = 0.30
VSCORE_THRESHOLD_ELEVATED = 0.50
VSCORE_THRESHOLD_HIGH = 0.70
VSCORE_THRESHOLD_CRITICAL = 0.85
TRIGGER_EXTERNAL_FACTCHECK = 0.85  # V-Score threshold
TRIGGER_LIMIT_SPREAD = 0.70
FACTCHECK_LATENCY_NORMAL = 3
FACTCHECK_LATENCY_URGENT = 1
FACTCHECK_BASE_ACCURACY = 0.90
ALERT_INFECTION_RATE = 0.30     # Alert if infection > 30%
ALERT_LOW_MODERATION = 0.20     # Alert if moderation < 20%
ALERT_HIGH_DISINFO_REP = 0.70   # Alert if disinformant rep > 70%
GENERATE_HTML_REPORT = True
GENERATE_JSON_HISTORY = True
GENERATE_PNG_DASHBOARD = True
ENABLE_THEORY_OF_MIND = True        # Agents consider what neighbors think
ENABLE_EMOTIONAL_VOLATILITY = True  # Emotions affect decisions
ENABLE_CONFIRMATION_BIAS = True     # Agents prefer confirming info
ENABLE_REPUTATION_CONCERN = True    # Agents care about social image
VALIDATE_STATE_COUNTS = True   # Verify population counts sum correctly
LOG_AGENT_DECISIONS = False    # Detailed decision logging (slow!)
LOG_STRATEGIC_MOVES = True     # Log strategic agent moves
USE_NUMPY_VECTORIZATION = True  # Faster computations where possible
