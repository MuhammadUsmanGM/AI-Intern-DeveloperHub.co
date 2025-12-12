# AI/ML Engineering Internship Portfolio

This repository contains the complete set of AI/ML projects completed as part of the DevelopersHub AI/ML Engineering Internship program.

## Live Deployments

Access the applications online:

- **Task 1 - Iris Dataset Explorer**: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-intern-developerappco-7fhcksrtr4zum3jnyh9eht.streamlit.app/)
- **Task 2 - Stock Price Predictor**: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-intern-developerappco-d4nxyuajpm9z3dfheryxyr.streamlit.app/)
- **Task 3 - Heart Disease Predictor**: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-intern-developerappco-2gyhhlkeeadrrrycw2h9bp.streamlit.app/)
- **Task 4 - Health Query Assistant**: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-intern-developerappco-ev9yfkshr9zdh3xsvyyacq.streamlit.app/)
- **Task 5 - Mental Health Chatbot**: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-intern-developerappco-tsfmow8wgxuccc53kzvkje.streamlit.app/)
- **Task 6 - House Price Predictor**: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-intern-developerappco-6b4gz4tvvfxs5rkvgpzf3e.streamlit.app/)

## Portfolio Dashboard

The `portfolio_dashboard_app.py` file contains a Streamlit application that combines all the individual projects into a single interface. This allows you to navigate between different applications from one dashboard.

To run the portfolio dashboard:
```bash
streamlit run portfolio_dashboard_app.py
```

## Individual Projects

Each project is organized in its respective task folder:

- **Task 1**: Iris Dataset Exploration
- **Task 2**: Stock Price Prediction
- **Task 3**: Heart Disease Prediction (includes heart_disease_predictor_app.py)
- **Task 4**: General Health Chatbot (includes health_chatbot_app.py)
- **Task 5**: Mental Health Chatbot (includes mental_health_chatbot_app.py)
- **Task 6**: House Price Prediction (includes house_price_predictor_app.py)

Each task folder contains the Jupyter notebook, README, and any associated Streamlit applications.

## Requirements

Each task has its own `requirements.txt` file in its respective folder. Install requirements for each task separately:
```bash
# For Task 1 (Iris Dataset Explorer)
cd Task1-Irsi_Dataset_Exploration
pip install -r requirements.txt

# For Task 2 (Stock Price Prediction)
cd Task2-Stock_Price_Prediction
pip install -r requirements.txt

# For Task 3 (Heart Disease Prediction)
cd Task3-Heart_disease_Prediction
pip install -r requirements.txt

# For Task 4 (Health Query Assistant) - includes Google Generative AI
cd Task4_General_Health_Chatbot
pip install -r requirements.txt

# For Task 5 (Mental Health Chatbot) - includes Google Generative AI
cd Task5-Mental_Health_Chatbot_Finetune
pip install -r requirements.txt

# For Task 6 (House Price Prediction)
cd Task6-House_Price_Prediction
pip install -r requirements.txt
```

## Submission

This portfolio demonstrates completion of more than the required 3 out of 6 tasks, showcasing skills in data science, machine learning, NLP, and web application development.