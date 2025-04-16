import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Union
import time
import hashlib
import json


class ChallengeGenerator:
    """
    Generator for voice challenges in the challenge-response protocol.
    """
    
    def __init__(
        self, 
        challenge_types: Optional[List[str]] = None,
        challenge_difficulty: str = 'medium',
        session_expiry: int = 120  # seconds
    ):
        """
        Initialize the challenge generator.
        
        Args:
            challenge_types: Types of challenges to use
            challenge_difficulty: Difficulty level ('easy', 'medium', 'hard')
            session_expiry: Time in seconds after which a challenge expires
        """
        self.challenge_types = challenge_types or [
            'digit_sequence', 
            'phonetic_phrase', 
            'prompt_repeat', 
            'arithmetic'
        ]
        self.challenge_difficulty = challenge_difficulty
        self.session_expiry = session_expiry
        
        # Load challenge templates
        self.digit_templates = self._load_digit_templates()
        self.phonetic_templates = self._load_phonetic_templates()
        self.prompt_templates = self._load_prompt_templates()
        
        # Dictionary to store active challenges by session ID
        self.active_challenges = {}
    
    def generate_challenge(self, session_id: str = None) -> Dict:
        """
        Generate a new challenge for the user.
        
        Args:
            session_id: Optional session ID (generated if not provided)
            
        Returns:
            Dictionary with challenge details
        """
        # Generate session ID if not provided
        if session_id is None:
            session_id = self._generate_session_id()
        
        # Select challenge type
        challenge_type = random.choice(self.challenge_types)
        
        # Generate challenge based on type
        if challenge_type == 'digit_sequence':
            challenge = self._generate_digit_challenge()
        elif challenge_type == 'phonetic_phrase':
            challenge = self._generate_phonetic_challenge()
        elif challenge_type == 'prompt_repeat':
            challenge = self._generate_prompt_challenge()
        elif challenge_type == 'arithmetic':
            challenge = self._generate_arithmetic_challenge()
        else:
            raise ValueError(f"Unknown challenge type: {challenge_type}")
        
        # Create challenge object
        challenge_obj = {
            'session_id': session_id,
            'challenge_type': challenge_type,
            'challenge_text': challenge['text'],
            'challenge_display': challenge.get('display', challenge['text']),
            'expected_response': challenge['expected_response'],
            'difficulty': self.challenge_difficulty,
            'created_at': int(time.time()),
            'expires_at': int(time.time()) + self.session_expiry,
            'verification_hash': self._generate_verification_hash(
                session_id, challenge['expected_response']
            )
        }
        
        # Store in active challenges
        self.active_challenges[session_id] = challenge_obj
        
        # Return challenge without expected response
        challenge_for_user = challenge_obj.copy()
        challenge_for_user.pop('expected_response')
        
        return challenge_for_user
    
    def verify_response(
        self, 
        session_id: str, 
        response_text: str
    ) -> Dict:
        """
        Verify a user's response to a challenge.
        
        Args:
            session_id: Session ID of the challenge
            response_text: User's response text
            
        Returns:
            Dictionary with verification results
        """
        # Check if session exists
        if session_id not in self.active_challenges:
            return {
                'verified': False,
                'error': 'Invalid session ID',
                'session_id': session_id
            }
        
        # Get challenge
        challenge = self.active_challenges[session_id]
        
        # Check if expired
        current_time = int(time.time())
        if current_time > challenge['expires_at']:
            return {
                'verified': False,
                'error': 'Challenge expired',
                'session_id': session_id
            }
        
        # Check response
        expected = challenge['expected_response'].lower()
        actual = response_text.lower()
        
        # Normalize responses for comparison
        expected_norm = self._normalize_text(expected)
        actual_norm = self._normalize_text(actual)
        
        # Calculate similarity
        similarity = self._calculate_text_similarity(expected_norm, actual_norm)
        
        # Determine verification threshold based on difficulty
        if self.challenge_difficulty == 'easy':
            threshold = 0.7
        elif self.challenge_difficulty == 'medium':
            threshold = 0.8
        else:  # hard
            threshold = 0.9
        
        # Determine verification result
        verified = similarity >= threshold
        
        # Prepare result
        result = {
            'verified': verified,
            'similarity': similarity,
            'threshold': threshold,
            'session_id': session_id,
            'challenge_type': challenge['challenge_type'],
            'response_received': response_text,
            'verification_time': current_time
        }
        
        # Remove from active challenges if verified
        if verified:
            self.active_challenges.pop(session_id)
        
        return result
    
    def _generate_session_id(self) -> str:
        """
        Generate a unique session ID.
        
        Returns:
            Session ID string
        """
        # Create a unique ID based on time and random number
        unique_string = f"{time.time()}_{random.randint(1000, 9999)}"
        
        # Hash to create session ID
        session_id = hashlib.sha256(unique_string.encode()).hexdigest()[:16]
        
        return session_id
    
    def _generate_verification_hash(self, session_id: str, expected_response: str) -> str:
        """
        Generate a verification hash for challenge integrity.
        
        Args:
            session_id: Session ID
            expected_response: Expected response
            
        Returns:
            Verification hash
        """
        # Combine session ID and expected response
        to_hash = f"{session_id}:{expected_response}:{time.time()}"
        
        # Generate hash
        verification_hash = hashlib.sha256(to_hash.encode()).hexdigest()
        
        return verification_hash
    
    def _load_digit_templates(self) -> List[Dict]:
        """
        Load digit sequence challenge templates.
        
        Returns:
            List of digit sequence templates
        """
        return [
            {
                'format': 'Say the digits: {digits}',
                'min_digits': 4,
                'max_digits': 10,
                'allow_repetition': True
            },
            {
                'format': 'Please read this number: {digits}',
                'min_digits': 4,
                'max_digits': 10,
                'allow_repetition': True
            },
            {
                'format': 'Speak the following numbers in order: {digits}',
                'min_digits': 5,
                'max_digits': 8,
                'allow_repetition': False
            }
        ]
    
    def _load_phonetic_templates(self) -> List[Dict]:
        """
        Load phonetic phrase challenge templates.
        
        Returns:
            List of phonetic phrase templates
        """
        return [
            {
                'phrases': [
                    "The quick brown fox jumps over the lazy dog",
                    "She sells seashells by the seashore",
                    "How much wood would a woodchuck chuck",
                    "Peter Piper picked a peck of pickled peppers",
                    "Unique New York unique New York",
                    "Red leather yellow leather",
                    "The early bird catches the worm",
                    "A stitch in time saves nine",
                    "All that glitters is not gold",
                    "The pen is mightier than the sword"
                ]
            }
        ]
    
    def _load_prompt_templates(self) -> List[Dict]:
        """
        Load prompt repeat challenge templates.
        
        Returns:
            List of prompt repeat templates
        """
        return [
            {
                'prompts': [
                    "My voice is my passport",
                    "Verify me by my voice today",
                    "Access granted through voice recognition",
                    "Authentication is active now",
                    "Confirm my identity with these words",
                    "Voice biometrics in progress",
                    "Secure access with my voice print",
                    "Validation of my unique voice",
                    "This is a voice verification test",
                    "Proving my identity vocally"
                ]
            }
        ]
    
    def _generate_digit_challenge(self) -> Dict:
        """
        Generate a digit sequence challenge.
        
        Returns:
            Dictionary with challenge details
        """
        # Select template
        template = random.choice(self.digit_templates)
        
        # Determine number of digits based on difficulty
        if self.challenge_difficulty == 'easy':
            num_digits = random.randint(template['min_digits'], template['min_digits'] + 2)
        elif self.challenge_difficulty == 'medium':
            num_digits = random.randint(template['min_digits'] + 1, template['max_digits'] - 1)
        else:  # hard
            num_digits = template['max_digits']
        
        # Generate digit sequence
        if template['allow_repetition']:
            digits = [str(random.randint(0, 9)) for _ in range(num_digits)]
        else:
            # Without repetition
            digits = random.sample([str(i) for i in range(10)], min(num_digits, 10))
        
        # Format for display
        if num_digits > 5:
            # Group digits for easier reading
            display_digits = ' '.join([''.join(digits[i:i+3]) for i in range(0, len(digits), 3)])
        else:
            display_digits = ' '.join(digits)
        
        # Create challenge text
        challenge_text = template['format'].format(digits=display_digits)
        
        # Expected response is just the digits
        expected_response = ''.join(digits)
        
        return {
            'text': challenge_text,
            'display': challenge_text,
            'expected_response': expected_response
        }
    
    def _generate_phonetic_challenge(self) -> Dict:
        """
        Generate a phonetic phrase challenge.
        
        Returns:
            Dictionary with challenge details
        """
        # Select template
        template = random.choice(self.phonetic_templates)
        
        # Select phrase
        phrase = random.choice(template['phrases'])
        
        # Create challenge text
        challenge_text = f"Please say: '{phrase}'"
        
        return {
            'text': challenge_text,
            'display': challenge_text,
            'expected_response': phrase
        }
    
    def _generate_prompt_challenge(self) -> Dict:
        """
        Generate a prompt repeat challenge.
        
        Returns:
            Dictionary with challenge details
        """
        # Select template
        template = random.choice(self.prompt_templates)
        
        # Select prompt
        prompt = random.choice(template['prompts'])
        
        # Create challenge text
        challenge_text = f"Repeat after me: '{prompt}'"
        
        return {
            'text': challenge_text,
            'display': challenge_text,
            'expected_response': prompt
        }
    
    def _generate_arithmetic_challenge(self) -> Dict:
        """
        Generate a simple arithmetic challenge.
        
        Returns:
            Dictionary with challenge details
        """
        # Determine difficulty level
        if self.challenge_difficulty == 'easy':
            # Single-digit addition
            a = random.randint(1, 9)
            b = random.randint(1, 9)
            operator = '+'
            result = a + b
        elif self.challenge_difficulty == 'medium':
            # Two-digit addition or single-digit multiplication
            op_type = random.choice(['add', 'multiply'])
            if op_type == 'add':
                a = random.randint(10, 50)
                b = random.randint(10, 50)
                operator = '+'
                result = a + b
            else:
                a = random.randint(2, 9)
                b = random.randint(2, 9)
                operator = '×'
                result = a * b
        else:  # hard
            # Two-digit addition, subtraction, or multiplication
            op_type = random.choice(['add', 'subtract', 'multiply'])
            if op_type == 'add':
                a = random.randint(10, 99)
                b = random.randint(10, 99)
                operator = '+'
                result = a + b
            elif op_type == 'subtract':
                a = random.randint(50, 99)
                b = random.randint(10, a-1)  # Ensure positive result
                operator = '-'
                result = a - b
            else:
                a = random.randint(5, 15)
                b = random.randint(5, 15)
                operator = '×'
                result = a * b
        
        # Create challenge text
        challenge_text = f"Calculate and say the answer: {a} {operator} {b}"
        display_text = f"{a} {operator} {b} = ?"
        
        # Expected response is the result as text
        expected_response = str(result)
        
        return {
            'text': challenge_text,
            'display': display_text,
            'expected_response': expected_response
        }
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Remove punctuation and extra whitespace
        translator = str.maketrans('', '', '.,;:!?-')
        normalized = text.translate(translator)
        normalized = ' '.join(normalized.split())
        
        return normalized.lower()
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple calculation based on character-level edit distance
        # For more sophisticated comparison, consider using NLTK or similar
        from difflib import SequenceMatcher
        
        return SequenceMatcher(None, text1, text2).ratio()
