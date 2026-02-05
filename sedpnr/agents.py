# agents.py
"""
Human Like Behavior Influenced from custom Survey of 712 + people
"""

from mesa import Agent
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import config as cfg
import advanced_math as math_engine


class EmotionalState(Enum):
    """Current emotional disposition of an agent."""
    CALM = "calm"
    ANXIOUS = "anxious"
    FEARFUL = "fearful"
    ANGRY = "angry"
    CURIOUS = "curious"
    SKEPTICAL = "skeptical"
    TRUSTING = "trusting"

@dataclass
class CognitiveState:
    # Core emotions (0-1 intensity)
    fear: float = 0.2
    uncertainty: float = 0.5
    trust: float = 0.5
    anger: float = 0.1
    # Cognitive load
    attention_remaining: float = 1.0  # Depletes with exposure
    processing_depth: float = 0.5     # How carefully analyzing
    # Social state
    perceived_reputation: float = 0.5  # How I think others see me
    social_pressure: float = 0.0       # Pressure from neighbors
    # Memory
    exposures_today: int = 0
    shares_today: int = 0
    debunks_heard: int = 0
    
    def get_emotional_state(self) -> EmotionalState:
        if self.fear > 0.7:
            return EmotionalState.FEARFUL
        elif self.anger > 0.6:
            return EmotionalState.ANGRY
        elif self.uncertainty > 0.7:
            return EmotionalState.ANXIOUS
        elif self.trust < 0.3:
            return EmotionalState.SKEPTICAL
        elif self.trust > 0.7:
            return EmotionalState.TRUSTING
        elif self.processing_depth > 0.6:
            return EmotionalState.CURIOUS
        else:
            return EmotionalState.CALM
    
    def decay_attention(self, amount: float = 0.05):
        self.attention_remaining = max(0.1, self.attention_remaining - amount)
    
    def reset_daily(self):
        self.exposures_today = 0
        self.shares_today = 0
        self.attention_remaining = min(1.0, self.attention_remaining + 0.3)

class FractalConsumer(Agent):
    """
    SEDPNR Model 
    States: S (Susceptible), E (Exposed), D (Deciding), 
            P (Procrastinating/Spreader), N (Neutral), R (Recovered)
    
    Key innovations:
    1. Emotional reactions affect decision-making
    2. Theory of Mind: Considers "what will others think?"
    3. Confirmation bias: Seeks information that confirms existing beliefs
    4. Reputation management: Protects social standing
    5. Memory effects: Past exposures influence current reactions
    """
    
    def __init__(self, node_id, model, profile):
        super().__init__(model)
        self.node_id = node_id 
        self.profile = profile
        if hasattr(profile, 'openness'):
            self.traits = {
                'openness': profile.openness,
                'conscientiousness': profile.conscientiousness,
                'extraversion': profile.extraversion,
                'agreeableness': profile.agreeableness,
                'neuroticism': profile.neuroticism,
                'fear_sensitivity': getattr(profile, 'fear_sensitivity', 0.5),
                'uncertainty_tolerance': getattr(profile, 'uncertainty_tolerance', 0.5),
                'trust_baseline': getattr(profile, 'trust_baseline', 0.5),
                'social_conformity': getattr(profile, 'social_conformity', 0.5),
                'critical_thinking': getattr(profile, 'critical_thinking', 0.5),
                'emotional_volatility': getattr(profile, 'emotional_volatility', 0.5),
                'influence_susceptibility': getattr(profile, 'influence_susceptibility', 0.5),
                'sharing_propensity': getattr(profile, 'sharing_propensity', 0.5),
                'confirmation_bias': getattr(profile, 'confirmation_bias', 0.5),
                'reputation_concern': getattr(profile, 'reputation_concern', 0.5),
            }
        else:
            self.traits = {
                'openness': profile.get('openness', 0.5),
                'conscientiousness': profile.get('conscientiousness', 0.5),
                'extraversion': profile.get('extraversion', 0.5),
                'agreeableness': profile.get('agreeableness', 0.5),
                'neuroticism': profile.get('neuroticism', 0.5),
                'fear_sensitivity': profile.get('fear_sensitivity', 0.5),
                'uncertainty_tolerance': profile.get('uncertainty_tolerance', 0.5),
                'trust_baseline': profile.get('trust_baseline', 0.5),
                'social_conformity': profile.get('social_conformity', 0.5),
                'critical_thinking': profile.get('critical_thinking', 0.5),
                'emotional_volatility': profile.get('emotional_volatility', 0.5),
                'influence_susceptibility': profile.get('influence_susceptibility', 0.5),
                'sharing_propensity': profile.get('sharing_propensity', 0.5),
                'confirmation_bias': profile.get('confirmation_bias', 0.5),
                'reputation_concern': profile.get('reputation_concern', 0.5),
            }
        self.state = "S"
        self.time_in_state = 0
        self.belief = cfg.PRIOR_TRUTH_PROB
        self.cognitive_state = CognitiveState(
            trust=self.traits['trust_baseline'],
            uncertainty=0.5,
            fear=self.traits['fear_sensitivity'] * 0.3,
        )
        self._calculate_personal_parameters()
        self.recent_decisions: List[str] = []
        self.last_share_step: int = -100
        
    def _calculate_personal_parameters(self):
        """Calculate personalized transition rates from personality."""
        # Beta (Susceptibility to infection)
        raw_beta = cfg.BETA_INTERCEPT + \
                   (cfg.BETA_WEIGHTS.get('openness', 0.1) * self.traits['openness']) + \
                   (cfg.BETA_WEIGHTS.get('conscientiousness', -0.1) * self.traits['conscientiousness']) + \
                   (cfg.BETA_WEIGHTS.get('neuroticism', 0.1) * self.traits['neuroticism'])
        self.beta = max(0.05, min(0.95, raw_beta))
        
        # Lambda (Decision/processing rate)
        raw_lambda = cfg.LAMBDA_INTERCEPT + \
                     (cfg.LAMBDA_WEIGHTS.get('conscientiousness', 0.1) * self.traits['conscientiousness']) + \
                     (cfg.LAMBDA_WEIGHTS.get('openness', 0.05) * self.traits['openness'])
        self.decision_rate = max(0.05, min(0.95, raw_lambda))
        
        # Sharing threshold (lower = more likely to share)
        self.share_threshold = 0.3 + (0.4 * self.traits['conscientiousness']) - \
                               (0.2 * self.traits['extraversion'])
        self.share_threshold = max(0.1, min(0.9, self.share_threshold))
        
        # Recovery rate (how quickly they move past misinformation)
        self.recovery_rate = 0.1 + (0.3 * self.traits['critical_thinking'])
    
    def step(self):
        """Main agent step - cognitive state machine."""
        self.time_in_state += 1
        # Update emotional state based on environment
        self._update_emotional_state()
        # State machine transitions
        if self.state == "S":
            self._step_susceptible()
        elif self.state == "E":
            self._step_exposed()
        elif self.state == "D":
            self._step_deciding()
        elif self.state == "P":
            self._step_spreading()
        elif self.state == "N":
            self._step_neutral()
        # R (Recovered) is absorbing
    
    def _update_emotional_state(self):
        """Update emotional state based on environment and neighbors."""
        cs = self.cognitive_state
        
        # Social pressure from neighbors
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        if neighbors:
            spreading_neighbors = len([n for n in neighbors 
                                      if isinstance(n, FractalConsumer) and n.state == "P"])
            recovered_neighbors = len([n for n in neighbors 
                                       if isinstance(n, FractalConsumer) and n.state == "R"])
            
            # Pressure increases with more spreaders around
            cs.social_pressure = spreading_neighbors / len(neighbors)
            
            # Trust decreases if surrounded by spreaders
            if spreading_neighbors > recovered_neighbors:
                cs.trust = max(0.1, cs.trust - 0.02 * self.traits['social_conformity'])
        
        # Disinformant broadcast increases fear
        if self.model.disinformant.last_action == "share":
            fear_boost = 0.05 * self.traits['fear_sensitivity']
            cs.fear = min(1.0, cs.fear + fear_boost)
        
        # Fact-checker broadcast reduces fear
        if self.model.fact_checker.last_action == "debunk":
            fear_reduction = 0.1 * self.traits['critical_thinking']
            cs.fear = max(0.0, cs.fear - fear_reduction)
            cs.debunks_heard += 1
        
        # Natural emotion decay (regression to baseline)
        cs.fear = cs.fear * 0.95 + self.traits['fear_sensitivity'] * 0.05
        cs.anger = cs.anger * 0.9
    
    def _step_susceptible(self):
        """
        Susceptible State
        Agent is unaware of the misinformation.
        Transition to Exposed based on:
        - Neighbor infection pressure
        - Personal susceptibility
        - Disinformant broadcast
        """
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        if not neighbors:
            return
        # Calculate infection pressure from spreader neighbors
        infected_neighbors = [n for n in neighbors 
                             if isinstance(n, FractalConsumer) and n.state == "P"]
        base_force = len(infected_neighbors) / len(neighbors)
        # Boost from disinformant broadcast
        if self.model.disinformant.last_action == "share":
            # Disinformant's reputation affects persuasiveness
            disinfo_effect = 0.15 * self.model.disinformant.reputation
            base_force += disinfo_effect
        # Personal susceptibility modifier
        personal_susceptibility = (
            0.4 * self.traits['influence_susceptibility'] +
            0.3 * self.traits['fear_sensitivity'] +
            0.2 * self.cognitive_state.social_pressure +
            0.1 * (1 - self.traits['critical_thinking'])
        )
        # Fractional probability (memory effect - longer exposure = higher chance)
        prob = math_engine.fractional_probability(self.beta, self.time_in_state)
        # Combined probability
        infection_prob = base_force * prob * (0.5 + 0.5 * personal_susceptibility)
        if self.random.random() < infection_prob:
            self._transition_to("E")
            self.cognitive_state.exposures_today += 1
    def _step_exposed(self):
        """
        Exposed State
        Agent has seen the content but hasn't decided yet.
        Bayesian belief updating + emotional processing.
        """
        # Deplete attention with each step of exposure
        self.cognitive_state.decay_attention(0.03)
        
        # Bayesian belief update
        if self.model.disinformant.last_action == "share":
            # Update belief based on disinformant reputation
            self.belief = math_engine.bayesian_belief_update(
                self.belief,
                signal_is_share=True,
                rep_sender=self.model.disinformant.reputation
            )
            
            # Confirmation bias: If already leaning towards believing, amplify
            if self.belief < 0.4 and self.traits['confirmation_bias'] > 0.5:
                self.belief -= 0.05 * self.traits['confirmation_bias']
                self.belief = max(0.0, self.belief)
        
        # Emotional processing affects decision speed
        emotional_modifier = 1.0
        emotional_state = self.cognitive_state.get_emotional_state()
        
        if emotional_state == EmotionalState.FEARFUL:
            emotional_modifier = 1.5  # Fear speeds up decisions (fight-or-flight)
        elif emotional_state == EmotionalState.SKEPTICAL:
            emotional_modifier = 0.7  # Skepticism slows processing
        elif emotional_state == EmotionalState.CURIOUS:
            emotional_modifier = 1.2  # Curiosity motivates
        
        # Transition to Deciding state
        proc_prob = math_engine.fractional_probability(
            self.decision_rate * emotional_modifier,
            self.time_in_state
        )
        
        if self.random.random() < proc_prob:
            self._transition_to("D")
    
    def _step_deciding(self):
        """
        Deciding 
        Agent is actively deciding what to do.
        Uses Theory of Mind and reputation considerations.
        """
        # Apply Theory of Mind: "What will my neighbors think if I share this?"
        tom_modifier = self._theory_of_mind_check()
        
        # Neuroticism increases threshold volatility
        neuro_noise = (self.random.random() - 0.5) * 0.2 * self.traits['neuroticism']
        
        # Decision thresholds
        adjusted_low = cfg.THRESHOLD_LOW + neuro_noise + tom_modifier
        adjusted_high = cfg.THRESHOLD_HIGH - neuro_noise
        
        # Emotional override: High fear can push towards sharing (warning others)
        if self.cognitive_state.fear > 0.7:
            adjusted_low -= 0.1  # Makes it easier to become a spreader
        
        # Social pressure effect
        pressure_effect = 0.1 * self.cognitive_state.social_pressure * self.traits['social_conformity']
        adjusted_low -= pressure_effect  # Pressure makes spreading easier
        
        # Make decision
        if self.belief < adjusted_low:
            # Believe the misinformation -> become spreader
            self._transition_to("P")
            self.cognitive_state.shares_today += 1
            self.recent_decisions.append("share")
        elif self.belief > adjusted_high:
            # Believe it's false -> recover (reject)
            self._transition_to("R")
            self.recent_decisions.append("reject")
        else:
            # Uncertain -> become neutral
            self._transition_to("N")
            self.recent_decisions.append("ignore")
        
        # Trim decision history
        self.recent_decisions = self.recent_decisions[-10:]
    
    def _step_spreading(self):
        """
        Spreading
        Agent is actively spreading disinformation.
        Can recover through fact-checking or natural decay.
        """
        # Check for exposure to debunking
        if self.model.fact_checker.last_action == "debunk":
            # Probability of hearing and accepting the debunk
            # Higher critical thinking = more likely to accept correction
            hear_prob = 0.3 + (0.5 * self.traits['critical_thinking'])
            
            # Reputation of fact-checker matters
            hear_prob *= (0.5 + 0.5 * self.model.fact_checker.reputation)
            
            # Past debunks make new ones more credible
            repeated_exposure_boost = min(0.3, 0.1 * self.cognitive_state.debunks_heard)
            hear_prob += repeated_exposure_boost
            
            if self.random.random() < hear_prob:
                self._transition_to("R")
                self.belief = 0.7  # Corrected belief
                return
        
        # Reputation damage check: If neighbors are recovering, reconsider
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        if neighbors and self.traits['reputation_concern'] > 0.5:
            recovered_neighbors = len([n for n in neighbors 
                                       if isinstance(n, FractalConsumer) and n.state == "R"])
            if recovered_neighbors / len(neighbors) > 0.3:
                # Many neighbors recovered - feel social pressure to stop
                shame_prob = 0.1 * self.traits['reputation_concern']
                if self.random.random() < shame_prob:
                    self._transition_to("N")
                    return
        
        # Natural decay (fatigue, loss of interest)
        decay_prob = math_engine.fractional_probability(cfg.EPSILON_BASE, self.time_in_state)
        decay_prob *= (1 + 0.2 * self.traits['conscientiousness'])  # Conscientious tire faster
        
        if self.random.random() < decay_prob:
            self._transition_to("N")
    
    def _step_neutral(self):
        """
        Neutral
        Agent is disengaged but could be re-exposed.
        Slight chance of returning to susceptible or moving to recovered.
        """
        # Small chance of becoming fully recovered (moved on)
        if self.random.random() < 0.05:
            self._transition_to("R")
        
        # Can be re-infected if exposed again (but with resistance)
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        if neighbors:
            infected_neighbors = len([n for n in neighbors 
                                     if isinstance(n, FractalConsumer) and n.state == "P"])
            if infected_neighbors > 0:
                reinfection_prob = 0.02 * (infected_neighbors / len(neighbors))
                reinfection_prob *= self.traits['influence_susceptibility']
                if self.random.random() < reinfection_prob:
                    self._transition_to("E")
    
    def _theory_of_mind_check(self) -> float:
        """
        Theory of Mind 
        Agent estimates what neighbors will think if they share.
        Returns a modifier to the decision threshold.
        """
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        if not neighbors:
            return 0.0
        recovered_count = 0
        skeptical_count = 0
        
        for n in neighbors:
            if isinstance(n, FractalConsumer):
                if n.state == "R":
                    recovered_count += 1
                elif n.state == "S" and n.traits.get('critical_thinking', 0.5) > 0.6:
                    skeptical_count += 1
        social_risk = (recovered_count + skeptical_count * 0.5) / len(neighbors)
        tom_effect = social_risk * self.traits['reputation_concern'] * 0.2
        
        return tom_effect 
    
    def _transition_to(self, new_state: str):
        """Helper to transition states cleanly."""
        self.state = new_state
        self.time_in_state = 0

class DisinformantAgent(Agent):
    
    def __init__(self, model):
        super().__init__(model)
        self.reputation = 0.5  
        self.last_action = "hold"
        
        # Strategic memory
        self.successful_shares = 0
        self.caught_count = 0
        self.consecutive_holds = 0
        self.strategy_mode = "opportunistic"  # or "aggressive", "cautious"
        
    def step(self):
        """Strategic decision-making for the disinformant."""
        # 1. Assess the environment
        s_count = self.model.get_state_count("S")
        p_count = self.model.get_state_count("P")
        r_count = self.model.get_state_count("R")
        mod_level = self.model.platform.moderation_level
        
        # 2. Estimate opportunity
        # More susceptible agents = higher potential spread
        susceptible_ratio = s_count / max(1, self.model.num_agents)
        
        # Estimate spread potential
        estimated_spread = s_count * 0.15 * (1 - mod_level) * self.reputation
        
        # 3. Calculate risk
        # Risk increases with moderation and decreases with reputation
        detection_risk = mod_level * (1.5 - self.reputation)
        
        # Fact-checker activity increases risk
        if self.model.fact_checker.last_action == "debunk":
            detection_risk *= 1.3
        
        # 4. Calculate utility
        utility = math_engine.calculate_disinformant_utility(
            estimated_spread, mod_level, self.reputation
        )
        
        # 5. Apply strategic mode adjustments
        if self.strategy_mode == "aggressive":
            utility *= 1.2
        elif self.strategy_mode == "cautious":
            utility *= 0.8
        
        # 6. Consider timing: Build credibility during quiet periods
        if self.consecutive_holds > 5 and susceptible_ratio > 0.5:
            # Good time to strike
            utility += 0.5
        
        # 7. Make decision
        if utility > 0 and self.random.random() < (0.5 + utility * 0.3):
            self.last_action = "share"
            self.consecutive_holds = 0
            
            # Reputation grows slightly with successful shares
            if self.model.fact_checker.last_action != "debunk":
                self.successful_shares += 1
        else:
            self.last_action = "hold"
            self.consecutive_holds += 1
            
            # Reputation recovers during quiet periods
            self.reputation = min(1.0, self.reputation + 0.01)
        
        # 8. Adapt strategy based on success/failure ratio
        success_rate = self.successful_shares / max(1, self.successful_shares + self.caught_count)
        if success_rate > 0.7:
            self.strategy_mode = "aggressive"
        elif success_rate < 0.3:
            self.strategy_mode = "cautious"
        else:
            self.strategy_mode = "opportunistic"


class FactCheckerAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.reputation = 0.5  
        self.last_action = "wait"
        self.energy = 1.0  
        self.debunks_today = 0
        self.successful_corrections = 0
        
    def step(self):
        p_count = self.model.get_state_count("P")
        total = self.model.num_agents
        infection_rate = p_count / total
        if self.last_action == "wait":
            self.energy = min(1.0, self.energy + 0.1)
        urgency = 0.0
        if infection_rate > 0.05:
            urgency += (infection_rate - 0.05) * 5
        if self.model.disinformant.last_action == "share":
            urgency += 0.3
        urgency += 0.2 * self.model.platform.moderation_level
        if urgency > 0.3 and self.energy > 0.2:
            self.last_action = "debunk"
            self.energy -= 0.15
            self.debunks_today += 1
            if self.debunks_today < 5:
                self.reputation = min(1.0, self.reputation + 0.03)
            else:
                self.reputation = max(0.1, self.reputation - 0.01)
        else:
            self.last_action = "wait"
            if infection_rate > 0.2:
                self.reputation = max(0.1, self.reputation - 0.02)


class PlatformAgent(Agent):    
    def __init__(self, model):
        super().__init__(model)
        self.moderation_level = 0.5  # 0 = Lenient, 1 = Strict
        self.user_trust = 0.7  # Public trust in platform
        self.policy_mode = "balanced"  # "growth", "balanced", "safety"
        
        # Performance tracking
        self.engagement_history: List[int] = []
        self.misinfo_history: List[int] = []
        
        # Connection to defense systems (set by model)
        self.virality_oracle = None
        self.fact_check_router = None
        
    def step(self):
        """Stackelberg optimization for moderation level."""
        engagement = self.model.get_total_engagement()
        misinfo = self.model.get_state_count("P")
        susceptible = self.model.get_state_count("S")
        
        # Track history for trend analysis
        self.engagement_history.append(engagement)
        self.misinfo_history.append(misinfo)
        if len(self.engagement_history) > 20:
            self.engagement_history = self.engagement_history[-20:]
            self.misinfo_history = self.misinfo_history[-20:]
        
        # Calculate current utility
        curr_util = math_engine.platform_utility(engagement, self.moderation_level, misinfo)
        
        # Simulate alternative policies
        step_size = 0.05
        
        # Scenario A: Increase moderation
        projected_engagement_up = engagement * (0.95 - 0.05 * self.moderation_level)
        projected_misinfo_up = misinfo * 0.7
        util_up = math_engine.platform_utility(
            projected_engagement_up,
            min(1.0, self.moderation_level + step_size),
            projected_misinfo_up
        )
        
        # Scenario B: Decrease moderation  
        projected_engagement_down = engagement * 1.05
        projected_misinfo_down = misinfo * 1.2
        util_down = math_engine.platform_utility(
            projected_engagement_down,
            max(0.0, self.moderation_level - step_size),
            projected_misinfo_down
        )
        oracle_boost = 0.0
        if self.virality_oracle:
            pass
        if util_up > curr_util + oracle_boost and util_up > util_down:
            self.moderation_level = min(1.0, self.moderation_level + step_size)
        elif util_down > curr_util + oracle_boost:
            self.moderation_level = max(0.0, self.moderation_level - step_size)
        infection_rate = misinfo / max(1, self.model.num_agents)
        if infection_rate > 0.3:
            self.user_trust = max(0.2, self.user_trust - 0.02)
        elif infection_rate < 0.1 and self.moderation_level < 0.7:
            self.user_trust = min(1.0, self.user_trust + 0.01)
        if infection_rate > 0.25:
            self.policy_mode = "safety"
        elif engagement < self.model.num_agents * 0.3:
            self.policy_mode = "growth"
        else:
            self.policy_mode = "balanced"
    
    def trigger_defense_protocol(self, content_id: str, v_score: float):
        if v_score > 0.85:
            self.moderation_level = min(1.0, self.moderation_level + 0.2)
            
            # Route to external fact-checkers (not fully implemented yet - still in beta)
            if self.fact_check_router:
                self.fact_check_router.route_to_factchecker(
                    content_id=content_id,
                    content_description=f"Critical viral threat (V={v_score:.2f})",
                    v_score=v_score,
                    urgency="critical"
                )
        elif v_score > 0.7:
            # High threat - moderate response
            self.moderation_level = min(1.0, self.moderation_level + 0.1)
