# data_loader.py
"""
Survey Data - Agents
Loads survey data and analyses it to create an "accurate repersentation" of what humans would actually think. The questions included:
- Sharing behavior (instinctive vs. thoughtful)
- Verification habits
- Disinformation awareness
- Big Five personality traits (openness, conscientiousness, etc.)
"""
import pandas as pd
import numpy as np
import config as cfg
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import hashlib

LIKERT_MAP = {
    'Strongly disagree': 0.0,
    'Disagree': 0.25,
    'Neutral': 0.5,
    'Agree': 0.75,
    'Strongly Agree': 1.0,
    'Strongly agree': 1.0,  # Handle case variations
}

def likert_to_float(value) -> float:
    """Convert Likert scale text to 0-1 float."""
    if pd.isna(value):
        return 0.5
    value_str = str(value).strip()
    return LIKERT_MAP.get(value_str, 0.5)
@dataclass
class RealHumanProfile:
    # Identity
    agent_id: str
    age: str = "Unknown"
    gender: str = "Unknown"
    share_instinct: float = 0.5      # "When I see news that angers me, I share it"
    share_frequency: float = 0.5     # "I frequently share articles"
    verification_habit: float = 0.5  # "I verify from multiple sources"
    
    # Disinformation awareness
    disinfo_awareness: float = 0.5   # "Disinformation undermines..."
    influence_awareness: float = 0.5 # "Easy to be influenced by disinfo"
    verification_ease: float = 0.5   # "I find it easy to verify"
    factcheck_belief: float = 0.5    # "Reporting posts is effective"
    
    # Big Five personality (from self-assessment questions)
    openness: float = 0.5           # "Quick to understand, variety of activities"
    conscientiousness: float = 0.5  # "Organised, best of my ability"
    agreeableness: float = 0.5      # "Trusting and compassionate"
    extraversion: float = 0.5       # "Outgoing and sociable"
    neuroticism: float = 0.5        # "Gets nervous easily"
    
    # Value dimensions (from importance questions)
    hedonism: float = 0.5           # Enjoy life
    self_direction: float = 0.5     # Think differently
    achievement: float = 0.5        # Be successful
    tradition: float = 0.5          # Follow customs
    benevolence: float = 0.5        # Loyal to close ones
    stimulation: float = 0.5        # Try new things
    power: float = 0.5              # Influence over others
    conformity: float = 0.5         # Follow rules
    universalism: float = 0.5       # Care for all
    security: float = 0.5           # Safe and stable life
    social_platforms: str = ""
    follower_count: str = ""
    
    def get_susceptibility(self) -> float:
        return np.clip(
            0.25 * self.share_instinct +          # Shares impulsively
            0.20 * self.share_frequency +          # Shares often
            0.20 * (1 - self.verification_habit) + # Doesn't verify
            0.15 * (1 - self.disinfo_awareness) +  # Not aware of problem
            0.10 * self.neuroticism +              # Gets nervous (emotional)
            0.10 * (1 - self.conscientiousness),   # Not careful
            0.0, 1.0
        )
    def get_sharing_propensity(self) -> float:
        """How likely to share content (regardless of truth)."""
        return np.clip(
            0.35 * self.share_instinct +
            0.35 * self.share_frequency +
            0.15 * self.extraversion +
            0.15 * (1 - self.verification_habit),
            0.0, 1.0
        )
    
    def get_critical_thinking(self) -> float:
        """How likely to fact-check and reject false info."""
        return np.clip(
            0.30 * self.verification_habit +
            0.25 * self.verification_ease +
            0.20 * self.conscientiousness +
            0.15 * self.openness +
            0.10 * self.factcheck_belief,
            0.0, 1.0
        )
    
    def get_influence_reach(self) -> float:
        """Estimated reach based on social media presence."""
        follower_str = str(self.follower_count).lower()
        if '10000' in follower_str or '10,000' in follower_str or 'more' in follower_str:
            return 0.9
        elif '1000' in follower_str or '1,000' in follower_str:
            return 0.7
        elif '500' in follower_str:
            return 0.5
        elif '100' in follower_str:
            return 0.3
        else:
            return 0.2
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'agent_id': self.agent_id,
            'age': self.age,
            'gender': self.gender,
            'share_instinct': self.share_instinct,
            'share_frequency': self.share_frequency,
            'verification_habit': self.verification_habit,
            'susceptibility': self.get_susceptibility(),
            'sharing_propensity': self.get_sharing_propensity(),
            'critical_thinking': self.get_critical_thinking(),
            'openness': self.openness,
            'conscientiousness': self.conscientiousness,
            'agreeableness': self.agreeableness,
            'extraversion': self.extraversion,
            'neuroticism': self.neuroticism,
        }


def load_survey_data(filepath: str = None) -> List[RealHumanProfile]:
    filepath = filepath or cfg.DATA_FILENAME
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Survey data file '{filepath}' not found.")
    
    print(f"\n{'='*60}")
    print("LOADING REAL HUMAN SURVEY DATA")
    print(f"{'='*60}")
    print(f"Source: {filepath}")
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Could not decode CSV file")
    print(f"Loaded {len(df)} survey responses")
    
    profiles = []
    
    for idx, row in df.iterrows():
        agent_id = f"agent_{idx:04d}"
        cols = df.columns.tolist()
        profile = RealHumanProfile(
            agent_id=agent_id,
            age=str(row.iloc[4]) if len(cols) > 4 else "Unknown",
            gender=str(row.iloc[5]) if len(cols) > 5 else "Unknown",
            share_instinct=likert_to_float(row.iloc[8]) if len(cols) > 8 else 0.5,
            share_frequency=likert_to_float(row.iloc[9]) if len(cols) > 9 else 0.5,
            verification_habit=likert_to_float(row.iloc[10]) if len(cols) > 10 else 0.5,
            disinfo_awareness=likert_to_float(row.iloc[11]) if len(cols) > 11 else 0.5,
            influence_awareness=likert_to_float(row.iloc[12]) if len(cols) > 12 else 0.5,
            verification_ease=likert_to_float(row.iloc[13]) if len(cols) > 13 else 0.5,
            factcheck_belief=likert_to_float(row.iloc[14]) if len(cols) > 14 else 0.5,
            hedonism=likert_to_float(row.iloc[15]) if len(cols) > 15 else 0.5,
            self_direction=likert_to_float(row.iloc[16]) if len(cols) > 16 else 0.5,
            achievement=likert_to_float(row.iloc[17]) if len(cols) > 17 else 0.5,
            tradition=likert_to_float(row.iloc[18]) if len(cols) > 18 else 0.5,
            benevolence=likert_to_float(row.iloc[19]) if len(cols) > 19 else 0.5,
            stimulation=likert_to_float(row.iloc[20]) if len(cols) > 20 else 0.5,
            power=likert_to_float(row.iloc[21]) if len(cols) > 21 else 0.5,
            conformity=likert_to_float(row.iloc[22]) if len(cols) > 22 else 0.5,
            universalism=likert_to_float(row.iloc[23]) if len(cols) > 23 else 0.5,
            security=likert_to_float(row.iloc[24]) if len(cols) > 24 else 0.5,
            openness=likert_to_float(row.iloc[25]) if len(cols) > 25 else 0.5,
            conscientiousness=likert_to_float(row.iloc[26]) if len(cols) > 26 else 0.5,
            agreeableness=likert_to_float(row.iloc[27]) if len(cols) > 27 else 0.5,
            extraversion=likert_to_float(row.iloc[28]) if len(cols) > 28 else 0.5,
            neuroticism=likert_to_float(row.iloc[29]) if len(cols) > 29 else 0.5,
            social_platforms=str(row.iloc[6]) if len(cols) > 6 else "",
            follower_count=str(row.iloc[7]) if len(cols) > 7 else "",
        )
        profiles.append(profile)
    susceptibilities = [p.get_susceptibility() for p in profiles]
    sharing = [p.get_sharing_propensity() for p in profiles]
    critical = [p.get_critical_thinking() for p in profiles]
    
    print(f"\n Population Analysis")
    print(f"  Susceptibility - Mean: {np.mean(susceptibilities):.2f}, Std: {np.std(susceptibilities):.2f}")
    print(f"  Sharing Propensity - Mean: {np.mean(sharing):.2f}, Std: {np.std(sharing):.2f}")
    print(f"  Critical Thinking - Mean: {np.mean(critical):.2f}, Std: {np.std(critical):.2f}")
    high_susceptibility = [p for p in profiles if p.get_susceptibility() > 0.6]
    low_verification = [p for p in profiles if p.verification_habit < 0.4]
    
    print(f"\n Risk Indicators ")
    print(f"  High susceptibility (>0.6): {len(high_susceptibility)} agents ({100*len(high_susceptibility)/len(profiles):.1f}%)")
    print(f"  Low verification habit (<0.4): {len(low_verification)} agents ({100*len(low_verification)/len(profiles):.1f}%)")
    print(f"{'='*60}\n")
    
    return profiles
# For backward compatibility
def get_virality_susceptibility(profile: RealHumanProfile) -> float:
    """Get susceptibility score for oracle."""
    return profile.get_susceptibility()
