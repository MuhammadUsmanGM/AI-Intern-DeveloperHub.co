import streamlit as st
import numpy as np
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

class MentalHealthChatbot:
    """
    A supportive mental health chatbot with empathetic responses.
    """
    
    def __init__(self):
        self.conversation_history = []
        self.total_conversations = 0
        
        # Empathetic responses templates
        self.templates = {
            'anxiety': [
                "I understand anxiety can be overwhelming. Remember, you're not alone in feeling this way.",
                "That sounds really stressful. Have you tried any grounding techniques like deep breathing?",
                "Anxiety is your mind trying to protect you. Sometimes what we fear doesn't happen. Take it one moment at a time."
            ],
            'stress': [
                "Stress can really wear you down. It's important to take care of yourself during tough times.",
                "You're dealing with a lot. Have you considered talking to someone you trust about this?",
                "Remember, stress is temporary. You've gotten through hard times before, and you can do it again."
            ],
            'sad': [
                "It's okay to feel sad. Your emotions are valid and important.",
                "I'm sorry you're going through this. Sadness often means we care deeply about something.",
                "Difficult feelings do pass with time. Be gentle with yourself during this period."
            ],
            'lonely': [
                "Loneliness can be painful, but you're reaching out, which is a positive step.",
                "You deserve connection and support. Have you thought about joining a community?",
                "Feeling lonely doesn't mean something is wrong with you. Many people feel this way."
            ],
            'overwhelmed': [
                "When everything feels like too much, it helps to break things into smaller steps.",
                "You don't have to solve everything at once. What's one small thing you could do today?",
                "Feeling overwhelmed is a sign that you might need to slow down and prioritize."
            ],
            'default': [
                "I'm here to listen and support you. Thank you for sharing.",
                "Your feelings matter. I appreciate you opening up about this.",
                "It's brave of you to talk about what you're experiencing. What would help you most right now?"
            ]
        }
    
    def get_template_response(self, user_input: str) -> str:
        """Get a template-based empathetic response."""
        user_lower = user_input.lower()
        
        # Check for key emotional indicators
        for emotion, responses in self.templates.items():
            if emotion != 'default' and emotion in user_lower:
                return np.random.choice(responses)
        
        # Return default response
        return np.random.choice(self.templates['default'])
    
    def chat(self, user_input: str) -> str:
        """Main chat interface with conversation tracking."""
        response = self.get_template_response(user_input)
        
        # Track conversation
        self.conversation_history.append({
            'user': user_input,
            'bot': response
        })
        
        self.total_conversations += 1
        return response
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.conversation_history

# Set up the page configuration
st.set_page_config(
    page_title="🤗 Mental Health Support Bot",
    page_icon="🤗",
    layout="wide"
)

# Initialize chatbot
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = MentalHealthChatbot()

# Initialize message history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Title and description
st.title("🤗 Mental Health Support Chatbot")
st.markdown("""
This is a supportive chatbot trained on empathetic dialogues. 
Share what's on your mind, and I'll listen with care and understanding.
""")

# Display important disclaimers
st.warning("""
**IMPORTANT DISCLAIMER:** This chatbot provides SUPPORT and ENCOURAGEMENT ONLY. 
It is NOT a replacement for professional mental health treatment, therapy, or counseling. 
If you're in crisis, please reach out to emergency services or mental health professionals.
""")

# Crisis resources
with st.expander("🚨 Crisis Resources - Click here if you're in crisis", expanded=True):
    st.markdown("""
    **If you're in crisis, please contact:**
    - **National Suicide Prevention Lifeline**: 988 (call or text)
    - **Crisis Text Line**: Text HOME to 741741
    - **Emergency Services**: 911 (US) or your local emergency number
    - **International**: https://www.iasp.info/resources/Crisis_Centres/
    """)

# Display conversation history
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤗"):
            st.write(message["content"])

# User input
if prompt := st.chat_input("Share what's on your mind..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)
    
    # Generate bot response
    with st.chat_message("assistant", avatar="🤗"):
        response = st.session_state.chatbot.chat(prompt)
        st.write(response)
    
    # Store bot response
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with resources
with st.sidebar:
    st.header("💙 Mental Health Resources")
    
    st.markdown("""
    **When to Seek Professional Help:**
    - Persistent thoughts of self-harm or suicide
    - Severe anxiety or panic attacks
    - Depression lasting more than 2 weeks
    - Inability to function in daily life
    - Substance abuse concerns
    """)
    
    st.markdown("""
    **Helpful Practices:**
    - Regular exercise
    - Meditation and mindfulness
    - Social connections
    - Adequate sleep
    - Professional therapy
    """)
    
    st.markdown("---")
    st.info("Remember: Reaching out for professional help is a sign of strength, not weakness. You deserve proper care from qualified mental health professionals.")

# Footer
st.markdown("---")
st.markdown("Developed as part of AI/ML Engineering Internship Tasks")
st.markdown("This chatbot is designed to provide emotional support and general information, but cannot replace professional mental health services.")