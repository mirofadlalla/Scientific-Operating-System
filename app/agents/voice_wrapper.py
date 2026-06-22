"""
Voice-enabled wrapper for agents - adds speech synthesis capabilities
"""

from typing import Dict, Any
from app.audio import audio_processor


class VoiceEnabledAgent:
    """Wraps any agent to add voice output capabilities"""
    
    def __init__(self, agent):
        self.agent = agent
    
    async def run_and_speak(self, intent: str, entities: Dict[str, Any], voice: str = "nova") -> tuple:
        """
        Run agent and convert response to speech
        
        Returns:
            (text_response, audio_bytes): Text response and synthesized audio
        """
        # Get text response from agent
        text_response = await self.agent.run(intent, entities)
        
        # Synthesize to speech
        try:
            audio_bytes = await audio_processor.synthesize_speech(text_response, voice)
            return text_response, audio_bytes
        except Exception as e:
            print(f"[Voice synthesis failed] {str(e)}")
            return text_response, None
    
    async def run(self, intent: str, entities: Dict[str, Any]) -> str:
        """
        Regular agent run (just delegate)
        """
        return await self.agent.run(intent, entities)
