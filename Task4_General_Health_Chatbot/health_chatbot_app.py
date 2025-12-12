import streamlit as st
import re
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Define high-risk keywords that require emergency response
EMERGENCY_KEYWORDS = [
    'suicide', 'kill myself', 'harm myself', 'overdose',
    'choking', 'chest pain', 'heart attack', 'stroke',
    'severe bleeding', 'unconscious', 'unable to breathe',
    'poisoning', 'broken bone', 'severe injury'
]

# Define forbidden medical advice topics
FORBIDDEN_TOPICS = [
    'dosage', 'medication dose', 'drug dose',
    'perform surgery', 'surgery at home',
    'abortion', 'terminate pregnancy',
    'euthanasia', 'assisted suicide'
]

# Define valid health topics we can discuss
VALID_HEALTH_TOPICS = [
    'symptoms', 'causes', 'prevention', 'risk factors',
    'home remedies', 'when to see doctor', 'general information',
    'lifestyle changes', 'diet', 'exercise', 'sleep',
    'stress management', 'hygiene', 'first aid basics'
]

# System prompt for the health chatbot
SYSTEM_PROMPT = """You are a friendly, helpful, and responsible medical information assistant.

IMPORTANT GUIDELINES:
1. You provide GENERAL EDUCATIONAL INFORMATION about health topics, not medical diagnoses or treatment plans.
2. Always remind users to consult a healthcare professional for personal medical decisions.
3. NEVER provide specific medication doses or emergency instructions.
4. Be empathetic and supportive but maintain professional boundaries.
5. If unsure about something, recommend consulting a doctor.
6. Focus on prevention, healthy lifestyle, and when to seek professional help.

You respond in a clear, friendly, and easy-to-understand manner."""

# Safety response
SAFETY_RESPONSE = """I understand you may be in distress. Please:
- Contact emergency services immediately (911 in US, 112 in Europe, 999 in UK)
- Call your local poison control center
- Reach out to a mental health crisis line
- Tell a trusted family member or friend

I'm here to provide general health information, but this situation needs immediate professional help."""

# Forbidden response
FORBIDDEN_RESPONSE = """I can't provide specific advice on this topic. This is an area where you need professional medical consultation.

Please consult with:
- Your primary care physician
- A specialist in this medical field
- A licensed pharmacist (for medications)
- A mental health professional (for psychological concerns)

I'm happy to discuss general health information on other topics!"""

def is_emergency_query(text: str) -> bool:
    """Check if the query contains emergency-related keywords."""
    text_lower = text.lower()
    
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in text_lower:
            return True
    return False

def is_forbidden_topic(text: str) -> bool:
    """Check if the query asks for forbidden medical advice."""
    text_lower = text.lower()
    
    for topic in FORBIDDEN_TOPICS:
        if topic in text_lower:
            return True
    return False

def is_health_related(text: str) -> bool:
    """Check if the query is health-related."""
    health_keywords = [
        'health', 'disease', 'symptom', 'illness', 'pain', 'medicine',
        'doctor', 'hospital', 'treatment', 'cure', 'remedy', 'exercise',
        'diet', 'nutrition', 'sleep', 'stress', 'anxiety', 'depression',
        'cold', 'flu', 'fever', 'cough', 'headache', 'allergy',
        'blood pressure', 'diabetes', 'heart', 'infection'
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in health_keywords)

class HealthChatbot:
    """A responsible health information chatbot with safety filters and prompt engineering."""
    
    def __init__(self):
        self.conversation_history = []
        self.total_queries = 0
        self.blocked_queries = 0
    
    def check_safety(self, query: str) -> Dict[str, any]:
        """Check query for safety issues."""
        # Check for emergency
        if is_emergency_query(query):
            return {
                'is_safe': False,
                'reason': 'EMERGENCY_DETECTED',
                'response': SAFETY_RESPONSE,
                'severity': 'CRITICAL'
            }
        
        # Check for forbidden topics
        if is_forbidden_topic(query):
            return {
                'is_safe': False,
                'reason': 'FORBIDDEN_TOPIC',
                'response': FORBIDDEN_RESPONSE,
                'severity': 'HIGH'
            }
        
        # Check if it's health-related
        if not is_health_related(query):
            return {
                'is_safe': True,
                'reason': 'NON_HEALTH_QUERY',
                'response': None,
                'severity': 'NONE'
            }
        
        return {
            'is_safe': True,
            'reason': 'SAFE_QUERY',
            'response': None,
            'severity': 'NONE'
        }
    
    def _template_response(self, query: str) -> str:
        """Generate a template-based response for common health queries."""
        query_lower = query.lower()
        
        # Common health conditions
        if 'cold' in query_lower or 'cough' in query_lower:
            return (
                "Common cold is usually caused by viruses and resolves on its own. "
                "To help manage symptoms:\n"
                "• Rest and get plenty of sleep\n"
                "• Stay hydrated - drink water, tea, or broth\n"
                "• Use a humidifier to ease congestion\n"
                "• Gargle with salt water for sore throat\n"
                "• Use saline nasal drops for congestion\n\n"
                "Seek medical attention if symptoms persist beyond 10 days or worsen significantly."
            )
        
        elif 'headache' in query_lower:
            return (
                "Headaches can have various causes. General tips to manage them:\n"
                "• Rest in a quiet, dark room\n"
                "• Apply a cold or warm compress\n"
                "• Stay hydrated\n"
                "• Practice relaxation techniques\n"
                "• Avoid triggers like stress or certain foods\n\n"
                "If headaches are frequent or severe, consult a healthcare provider."
            )
        
        elif 'fever' in query_lower:
            return (
                "Fever is often the body's way of fighting infection. General care:\n"
                "• Rest and drink plenty of fluids\n"
                "• Keep the room cool and comfortable\n"
                "• Wear light clothing\n"
                "• Monitor your temperature\n\n"
                "Seek immediate medical attention if fever exceeds 103°F (39.4°C) or lasts more than 3 days."
            )
        
        elif 'sleep' in query_lower or 'insomnia' in query_lower:
            return (
                "Good sleep hygiene tips:\n"
                "• Maintain a consistent sleep schedule\n"
                "• Keep your bedroom cool, dark, and quiet\n"
                "• Avoid screens 30-60 minutes before bed\n"
                "• Limit caffeine, especially in the afternoon\n"
                "• Exercise regularly, but not close to bedtime\n"
                "• Try relaxation techniques like deep breathing\n\n"
                "If sleep problems persist, consult a sleep specialist."
            )
        
        elif 'stress' in query_lower or 'anxiety' in query_lower:
            return (
                "Ways to manage stress and anxiety:\n"
                "• Practice deep breathing exercises\n"
                "• Engage in regular exercise\n"
                "• Try meditation or mindfulness\n"
                "• Maintain social connections\n"
                "• Limit caffeine and alcohol\n"
                "• Ensure adequate sleep\n"
                "• Consider professional mental health support\n\n"
                "If anxiety significantly impacts daily life, please seek help from a mental health professional."
            )
        
        elif 'diet' in query_lower or 'nutrition' in query_lower:
            return (
                "Healthy eating guidelines:\n"
                "• Eat a variety of fruits and vegetables\n"
                "• Choose whole grains over refined grains\n"
                "• Include lean proteins in your diet\n"
                "• Limit sugar and processed foods\n"
                "• Stay hydrated with water\n"
                "• Eat in moderation\n\n"
                "For personalized nutrition advice, consult a registered dietitian."
            )
        
        elif 'exercise' in query_lower or 'workout' in query_lower:
            return (
                "Exercise recommendations:\n"
                "• Aim for at least 150 minutes of moderate aerobic activity per week\n"
                "• Include strength training 2-3 times per week\n"
                "• Start gradually if you're new to exercise\n"
                "• Choose activities you enjoy\n"
                "• Warm up before and cool down after exercise\n"
                "• Stay hydrated during exercise\n\n"
                "Consult your doctor before starting a new exercise program, especially if you have health concerns."
            )
        
        elif 'when' in query_lower and 'doctor' in query_lower:
            return (
                "You should see a doctor if you experience:\n"
                "• Symptoms that persist longer than expected\n"
                "• Severe pain or discomfort\n"
                "• Difficulty breathing or chest pain\n"
                "• Significant changes in your health\n"
                "• Concerns about your medications\n"
                "• Mental health concerns\n\n"
                "Don't delay seeking professional help if you're unsure - it's better to be cautious with your health!"
            )
        elif 'symptom' in query_lower and 'persist' in query_lower:
            return (
                "If your symptoms persist beyond the expected timeframe, it's important to consult with a healthcare professional.\n\n"
                "General guidelines:\n"
                "• Common cold symptoms: See a doctor if they last more than 10 days\n"
                "• Fever: Seek medical attention if it exceeds 103°F (39.4°C) or lasts more than 3 days\n"
                "• Headaches: Consult if they become frequent or severe\n"
                "• Any concerning symptoms: Better to be safe and get them checked\n\n"
                "Remember, only a qualified healthcare provider can properly evaluate persistent symptoms."
            )
        
        else:
            return (
                f"Thanks for your question about health. While I don't have a specific answer to '{query}', "
                "I recommend:\n"
                "• Consulting with a healthcare professional\n"
                "• Checking reliable health websites (WHO, CDC, Mayo Clinic)\n"
                "• Keeping track of your symptoms\n\n"
                "I'm here to provide general health information. Feel free to ask about symptoms, prevention, "
                "lifestyle changes, or when to see a doctor!"
            )
    
    def chat(self, query: str) -> str:
        """Main chat interface."""
        self.total_queries += 1
        
        # Check safety first
        safety_check = self.check_safety(query)
        
        # If not safe, return safety response
        if not safety_check['is_safe']:
            self.blocked_queries += 1
            return safety_check['response']
        
        # If not health-related, respond appropriately
        if safety_check['reason'] == 'NON_HEALTH_QUERY':
            return (
                "I'm specifically designed to help with health-related questions. "
                "Your question doesn't seem to be health-related. "
                "Feel free to ask me about symptoms, prevention, healthy habits, or when to see a doctor!"
            )
        
        # Generate template response
        response = self._template_response(query)
        
        # Store in history
        self.conversation_history.append({
            'user': query,
            'bot': response,
            'timestamp': len(self.conversation_history)
        })
        
        return response

# Set up the page configuration
st.set_page_config(
    page_title="🏥 Health Query Assistant",
    page_icon="🏥",
    layout="wide"
)

# Initialize chatbot
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = HealthChatbot()

# Initialize message history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Title and description
st.title("🏥 General Health Query Assistant")
st.markdown("""
This is a supportive health information assistant. 
Share your health-related question, and I'll provide general information and guidance.
""")

# Display disclaimer
st.warning("""
**IMPORTANT MEDICAL DISCLAIMER:** This chatbot provides GENERAL INFORMATION ONLY and is not a substitute for professional medical advice, diagnosis, or treatment. 
Always consult with qualified healthcare professionals for any medical concerns.
""")

# Crisis resources
with st.expander("⚠️ Crisis Resources - Click here if you're in emergency", expanded=False):
    st.markdown("""
    **If you're in crisis, please contact:**
    - **Emergency Services**: 911 (US), 999 (UK), 112 (EU)
    - **Suicide Prevention**: 988 (US)
    - **Crisis Text Line**: Text HOME to 741741
    """)

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if prompt := st.chat_input("Ask a health-related question..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate bot response
    with st.chat_message("assistant"):
        response = st.session_state.chatbot.chat(prompt)
        st.write(response)
    
    # Store bot response
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with health resources
with st.sidebar:
    st.header("📚 Health Resources")
    
    st.markdown("""
    **When to See a Doctor:**
    - Symptoms persist beyond expected timeframe
    - Severe pain or discomfort
    - Difficulty breathing or chest pain
    - Significant changes in health
    - Concerns about medications
    - Mental health concerns
    """)
    
    st.markdown("""
    **Healthy Living Tips:**
    - Eat a balanced diet
    - Exercise regularly
    - Get adequate sleep
    - Manage stress
    - Stay hydrated
    - Regular check-ups
    """)
    
    st.markdown("---")
    st.info("Remember: This chatbot provides general information only and should not replace professional medical advice.")

# Additional information
st.markdown("---")
st.markdown("Developed as part of AI/ML Engineering Internship Tasks")
st.markdown("This chatbot uses prompt engineering and safety filters to provide responsible health information.")