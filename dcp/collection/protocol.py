import os
import json
import numpy as np
import pandas as pd
import soundfile as sf
import random
import time
from datetime import datetime
import yaml
from .audio_acquisition import AudioAcquisitionSystem

class RecordingProtocol:
    def __init__(self, config_path="config/processing.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.acquisition_system = AudioAcquisitionSystem()
        self.session_data = {
            "session_id": None,
            "participant_id": None,
            "timestamp": None,
            "recordings": []
        }
        
        # Load script templates
        self.script_templates = self._load_script_templates()
        
    def _load_script_templates(self):
        """Load script templates for different recording types"""
        templates = {
            "phonetic_balanced": [
                "The quick brown fox jumps over the lazy dog.",
                "She sells seashells by the seashore.",
                "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
                "Peter Piper picked a peck of pickled peppers.",
                "Unique New York, unique New York, you know you need unique New York."
            ],
            "digits": [
                "Zero one two three four five six seven eight nine.",
                "Nine eight seven six five four three two one zero.",
                "Three one four one five nine two six five three five.",
                "Seven zero two four six eight zero zero one nine.",
                "Five five five one two three four five six seven."
            ],
            "command_phrases": [
                "Open the file.",
                "Call home.",
                "Set an alarm for seven A.M.",
                "What's the weather today?",
                "Play some music."
            ],
            "emotional": {
                "happy": "I'm so excited about this wonderful news!",
                "sad": "Unfortunately, I have to cancel our plans.",
                "angry": "This is absolutely unacceptable!",
                "neutral": "The train arrives at six thirty in the evening.",
                "surprised": "Wow! I didn't expect to see you here!"
            }
        }
        return templates
    
    def start_session(self, participant_id, session_type="standard"):
        """Start a new recording session"""
        self.session_data["session_id"] = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_data["participant_id"] = participant_id
        self.session_data["timestamp"] = datetime.now().isoformat()
        self.session_data["session_type"] = session_type
        self.session_data["recordings"] = []
        
        # Create session directory
        self.session_dir = os.path.join(
            "recordings", 
            f"participant_{participant_id}", 
            self.session_data["session_id"]
        )
        os.makedirs(self.session_dir, exist_ok=True)
        
        print(f"Started session {self.session_data['session_id']} for participant {participant_id}")
        return self.session_data["session_id"]
    
    def record_utterance(self, script_type, script_index=None, emotion=None, duration=None):
        """Record a single utterance following the protocol"""
        if self.session_data["session_id"] is None:
            raise ValueError("No active session. Call start_session first.")
        
        # Select script
        script = self._select_script(script_type, script_index, emotion)
        
        # Display script for participant
        print("\n" + "="*50)
        print(f"Please read the following text:")
        print(f"\n\"{script}\"\n")
        print("="*50)
        
        # Wait for participant to be ready
        input("Press Enter when ready to record...")
        
        # Start recording
        recording_id = f"{script_type}_{datetime.now().strftime('%H%M%S')}"
        output_path = os.path.join(self.session_dir, f"{recording_id}.wav")
        
        # Record
        self.acquisition_system.start_recording(output_path, duration)
        
        # If no duration specified, wait for user to stop
        if duration is None:
            input("Press Enter to stop recording...")
            self.acquisition_system.stop_recording()
        else:
            print(f"Recording for {duration} seconds...")
            time.sleep(duration)
            # The recording will stop automatically after the duration
        
        # Record metadata
        recording_data = {
            "recording_id": recording_id,
            "script_type": script_type,
            "script": script,
            "timestamp": datetime.now().isoformat(),
            "file_path": output_path,
            "emotion": emotion
        }
        
        self.session_data["recordings"].append(recording_data)
        print(f"Recorded utterance saved to {output_path}")
        
        return recording_data
    
    def _select_script(self, script_type, script_index=None, emotion=None):
        """Select appropriate script based on type and parameters"""
        if script_type == "emotional" and emotion:
            return self.script_templates["emotional"][emotion]
        
        templates = self.script_templates.get(script_type, self.script_templates["phonetic_balanced"])
        
        if script_index is not None and script_index < len(templates):
            return templates[script_index]
        else:
            return random.choice(templates)
    
    def run_standard_protocol(self, participant_id):
        """Run the standard recording protocol with all required utterances"""
        self.start_session(participant_id, "standard")
        
        # Phonetically balanced sentences
        for i in range(5):
            self.record_utterance("phonetic_balanced", i, duration=5)
        
        # Digits
        self.record_utterance("digits", 0, duration=5)
        
        # Command phrases
        for i in range(3):
            self.record_utterance("command_phrases", i, duration=3)
        
        # Emotional speech
        for emotion in ["neutral", "happy", "sad", "angry", "surprised"]:
            self.record_utterance("emotional", emotion=emotion, duration=5)
        
        # Save session metadata
        self.save_session_metadata()
        
        print(f"Completed standard protocol session {self.session_data['session_id']}")
        return self.session_data
    
    def run_multilingual_protocol(self, participant_id, languages=["english", "spanish"]):
        """Run multilingual recording protocol"""
        self.start_session(participant_id, "multilingual")
        
        # This would be extended with appropriate scripts for each language
        for language in languages:
            self.record_utterance("phonetic_balanced", 0, duration=5)
        
        # Save session metadata
        self.save_session_metadata()
        
        print(f"Completed multilingual protocol session {self.session_data['session_id']}")
        return self.session_data
    
    def save_session_metadata(self):
        """Save session metadata to JSON file"""
        metadata_path = os.path.join(self.session_dir, "session_metadata.json")
        
        with open(metadata_path, 'w') as f:
            json.dump(self.session_data, f, indent=2)
        
        print(f"Session metadata saved to {metadata_path}")
        return metadata_path
