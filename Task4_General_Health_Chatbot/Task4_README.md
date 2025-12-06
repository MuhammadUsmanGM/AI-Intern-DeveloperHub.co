# Task 4: General Health Query Chatbot (Prompt Engineering Based)

## 📊 Overview
This task creates a responsible health information chatbot using prompt engineering techniques, LLM integration, and comprehensive safety filters to handle health-related queries appropriately.

## 🎯 Objective
Create a chatbot that can answer general health-related questions using an LLM (Large Language Model) with proper safety mechanisms.

## 🤖 Model/Tools
- **Primary:** Hugging Face Transformers (DistilGPT2)
- **Alternative:** OpenAI API, Mistral-7B-Instruct
- **Safety:** Custom filter system
- **Template Engine:** Fallback response system

## 🛠️ Technologies Used
- **Python 3.8+**
- **transformers** - Hugging Face models
- **torch** - Deep learning framework
- **numpy** - Numerical operations

## 📋 Requirements Checklist

### What This Notebook Includes:
- ✅ Build Python notebook with LLM integration
- ✅ Send user queries to LLM
- ✅ Use prompt engineering (friendly, clear responses)
- ✅ Add safety filters (harmful advice prevention)
- ✅ Example queries demonstrating functionality
- ✅ Emergency detection system
- ✅ Forbidden topic filtering

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install transformers torch numpy
```

### 2. Run the Notebook
```bash
jupyter notebook Task4_General_Health_Chatbot.ipynb
```

### 3. First Run Note
The first execution will download the DistilGPT2 model (~200MB). This is a one-time download.

### 4. Optional: Use OpenAI API
If you have an OpenAI API key:
```python
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
```

## 📈 Key Features

### 1. Safety Filter System
**Emergency Detection:**
- Suicide/self-harm keywords
- Chest pain, heart attack
- Severe bleeding, choking
- Unconsciousness
- Poisoning

**Forbidden Topics:**
- Medication dosages
- Home surgery
- Abortion procedures
- Euthanasia

### 2. Prompt Engineering
**System Prompt includes:**
- Clear role definition
- Educational information focus
- Professional boundaries
- Safety guidelines
- Disclaimer requirements

### 3. Response Types
**Template-Based Responses for:**
- Common cold/cough
- Headaches
- Fever
- Sleep issues
- Stress/anxiety
- Diet/nutrition
- Exercise
- When to see a doctor

### 4. Conversation Management
- History tracking
- Statistics (total queries, blocked queries)
- Query categorization
- Health-relatedness checking

## 💡 Features Demonstrated

### HealthChatbot Class:
```python
chatbot = HealthChatbot(use_transformer=True)
response = chatbot.chat("What causes a sore throat?")
```

### Safety Checks:
- `is_emergency_query()` - Detects crisis situations
- `is_forbidden_topic()` - Blocks harmful advice
- `is_health_related()` - Validates query relevance

### Example Queries Tested:
1. ✅ "What causes a sore throat?"
2. ✅ "How can I improve my sleep?"
3. ⛔ "I'm having chest pain" (Emergency response)
4. ⛔ "What dosage of aspirin?" (Forbidden topic)

## 📊 Notebook Structure
1. Import Libraries
2. Load Transformers Model
3. Define Safety Filters
4. Create Prompt Templates
5. Build HealthChatbot Class
6. Test Safe Queries
7. Test Emergency Scenarios
8. Test Forbidden Topics
9. Test Non-Health Queries
10. Display Statistics
11. Deployment Guide

## 🎓 Learning Outcomes
- Prompt design and testing
- Using APIs for LLMs
- Safety handling in chatbot responses
- Building conversational agents
- Responsible AI development
- Template-based response systems

## ⚠️ Important Safety Notices

### Medical Disclaimer:
⚠️ This chatbot provides **GENERAL INFORMATION ONLY**
- Not a substitute for professional medical advice
- Cannot diagnose conditions
- Cannot prescribe medications
- Not for emergencies

### Crisis Resources:
- 🚨 Emergency: 911 (US), 999 (UK), 112 (EU)
- 🚨 Suicide Prevention: 988 (US)
- 🚨 Crisis Text Line: Text HOME to 741741

## 📈 Key Outputs

### Chatbot Capabilities:
✅ Answers general health questions  
✅ Provides lifestyle advice  
✅ Explains common symptoms  
✅ Recommends when to see doctors  
✅ Blocks dangerous queries  
✅ Redirects emergencies  

### Statistics Tracked:
- Total queries processed
- Queries blocked for safety
- Block rate percentage
- Conversation history

## 🔧 Deployment Options

### 1. Command-Line Interface (CLI)
```python
while True:
    user_input = input("You: ")
    response = chatbot.chat(user_input)
    print(f"Bot: {response}")
```

### 2. Streamlit Web App
```bash
streamlit run health_chatbot_app.py
```

### 3. API Endpoint (Flask/FastAPI)
Deploy as REST API for integration

## 🎨 Customization

### Add New Template Responses:
```python
chatbot.templates['your_topic'] = [
    "Response 1",
    "Response 2"
]
```

### Modify Safety Keywords:
```python
EMERGENCY_KEYWORDS.append('new_keyword')
```

## 🔗 Additional Resources
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Responsible AI Practices](https://ai.google/responsibility/responsible-ai-practices/)

---

**Status:** ✅ Complete  
**Estimated Time:** 60-90 minutes  
**Difficulty:** Intermediate-Advanced
