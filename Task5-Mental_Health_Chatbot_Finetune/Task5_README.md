# Task 5: Mental Health Support Chatbot (Fine-Tuned)

## 📊 Overview
This task fine-tunes a language model on empathetic dialogue data to create a supportive chatbot for mental health conversations, focusing on stress, anxiety, and emotional wellness.

## 🎯 Objective
Build a chatbot that provides supportive and empathetic responses for stress, anxiety, and emotional wellness using fine-tuned LLM.

## 🤖 Model Details
**Base Model:** DistilGPT2 (66M parameters)
- Lightweight and fast
- Easy to fine-tune on consumer hardware
- Good balance of quality and resource usage

**Alternative Models:**
- GPT-2 (124M parameters) - Better quality
- GPT-Neo (125M-2.7B parameters) - Higher quality
- Mistral 7B - Best quality (requires GPU)

## 📁 Dataset
**Empathetic Dialogues**
- **Source:** Facebook AI Research
- **Size:** 25k+ conversations
- **Content:** Human conversations with emotional context
- **Emotions Covered:** 32 emotion types
- **Use Case:** Training empathetic responses

**Dataset Link:** https://huggingface.co/datasets/empathetic_dialogues

## 🛠️ Technologies Used
- **Python 3.8+**
- **transformers** - Hugging Face library
- **torch** - Deep learning framework
- **datasets** - Hugging Face datasets
- **numpy** - Numerical operations
- **pandas** - Data handling

## 📋 Requirements Checklist

### What This Notebook Includes:
- ✅ Fine-tune DistilGPT2 using Trainer API
- ✅ Use EmpatheticDialogues dataset
- ✅ Train to respond empathetically
- ✅ Gentle and emotionally supportive tone
- ✅ Build Streamlit interface code
- ✅ Template-based fallback system
- ✅ Multi-turn conversation support

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install transformers torch datasets numpy pandas
```

### 2. Hardware Requirements
**Minimum:**
- CPU: Any modern processor
- RAM: 8GB
- Storage: 2GB free space

**Recommended:**
- GPU: NVIDIA GPU with 4GB+ VRAM
- RAM: 16GB
- Storage: 5GB free space

### 3. Run the Notebook
```bash
jupyter notebook Task5_Mental_Health_Chatbot_Finetune.ipynb
```

### 4. Training Time
- **CPU:** ~30-60 minutes (3 epochs)
- **GPU:** ~5-10 minutes (3 epochs)

## 📈 Key Features

### 1. Fine-Tuning Process
```python
# Automatic in notebook:
1. Load EmpatheticDialogues dataset
2. Preprocess and tokenize
3. Configure TrainingArguments
4. Fine-tune with Trainer API
5. Save fine-tuned model
```

### 2. MentalHealthChatbot Class
**Capabilities:**
- Template-based responses (immediate)
- Model-based responses (after fine-tuning)
- Conversation history tracking
- Multi-emotional scenario handling

**Emotional Categories:**
- Anxiety
- Stress
- Sadness
- Loneliness
- Overwhelm
- General support

### 3. Response Generation
```python
chatbot = MentalHealthChatbot(use_finetuned=True)
response = chatbot.chat("I've been feeling anxious lately")
```

## 💡 Features Demonstrated

### Training Configuration:
- **Epochs:** 3
- **Batch Size:** 4
- **Learning Rate:** 2e-5
- **Max Length:** 256 tokens
- **Optimizer:** AdamW

### Sample Conversations Tested:
1. ✅ "I've been feeling really anxious about my future"
2. ✅ "My best friend moved away and I'm lonely"
3. ✅ "Work has been overwhelming me"
4. ✅ "I failed my exam and feel like a failure"
5. ✅ "I haven't been sleeping well due to stress"

### Template Responses Include:
- Validation of emotions
- Empathetic acknowledgment
- Gentle suggestions
- Encouragement
- Professional help recommendations

## 📊 Notebook Structure
1. Import Libraries
2. Load EmpatheticDialogues Dataset
3. Explore Dataset Structure
4. Create Sample Training Data
5. Model & Tokenizer Setup
6. Prepare Training Data
7. Configure Fine-Tuning
8. Train Model
9. Save Fine-Tuned Model
10. Build MentalHealthChatbot Class
11. Test Scenarios
12. Streamlit Interface Code
13. Deployment Guide
14. Ethical Considerations

## 🎓 Learning Outcomes
- Model fine-tuning with Hugging Face Transformers
- Working with conversational datasets
- Emotional tone design for chatbots
- Conversation modeling
- Deployment using CLI or web apps
- Responsible AI for sensitive domains

## ⚠️ Critical Disclaimers

### Mental Health Notice:
⚠️ **This chatbot provides SUPPORT and ENCOURAGEMENT ONLY**

**NOT:**
- ❌ A replacement for therapy
- ❌ Capable of diagnosing conditions
- ❌ A crisis intervention service
- ❌ Medical or psychiatric treatment

**Crisis Resources:**
- 🚨 National Suicide Prevention Lifeline: **988**
- 🚨 Crisis Text Line: Text **HOME to 741741**
- 🚨 International: https://www.iasp.info/resources/Crisis_Centres/

## 🎨 Deployment

### 1. Streamlit Web App
```bash
# Code included in notebook
streamlit run mental_health_app.py
```

### 2. Command-Line Interface
```python
while True:
    user_input = input("Share what's on your mind: ")
    response = chatbot.chat(user_input)
    print(f"SupportiveBot: {response}")
```

### 3. Production Considerations
- ✅ HTTPS encryption
- ✅ Data privacy (HIPAA/GDPR)
- ✅ Crisis detection monitoring
- ✅ Regular model updates
- ✅ User feedback system

## 📈 Expected Results

### Fine-Tuned Model Quality:
- More empathetic than base model
- Context-aware responses
- Gentle and supportive tone
- Natural conversation flow

### Template System:
- Immediate fallback responses
- Covers 6+ emotional categories
- Professional and caring tone

## 🔧 Customization

### Add New Emotional Templates:
```python
chatbot.templates['your_emotion'] = [
    "Supportive response 1",
    "Supportive response 2"
]
```

### Adjust Training:
```python
training_args = TrainingArguments(
    num_train_epochs=5,  # More epochs
    learning_rate=3e-5,  # Adjust LR
)
```

## 🔗 Additional Resources
- [EmpatheticDialogues Paper](https://arxiv.org/abs/1811.00207)
- [Fine-Tuning Guide](https://huggingface.co/docs/transformers/training)
- [Mental Health AI Ethics](https://www.apa.org/monitor/2023/07/mental-health-ai)
- [Crisis Resources](https://988lifeline.org/)

---

**Status:** ✅ Complete  
**Estimated Time:** 90-120 minutes  
**Difficulty:** Advanced
