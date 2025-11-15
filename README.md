Telco Customer Churn Predictor with LIME and LLM Recommendations

This project is an interactive Streamlit application designed to predict customer churn in the telecommunications industry. It combines a trained machine learning model, LIME explainability, and a Large Language Model (LLM) to generate actionable customer retention strategies.

The system supports business decision-making by providing:

Churn predictions based on customer attributes

Clear explanations for each prediction

Intelligent recommendations for reducing churn

Features
1. Machine Learning Churn Prediction

The application uses a pre-trained model (telco_churn_model.pkl) to predict whether a customer is likely to churn. It also outputs the probability associated with the prediction.

2. LIME Explainability

LIME (Local Interpretable Model-Agnostic Explanations) is used to identify and visualize which features influenced the churn prediction. This helps stakeholders understand why the model reached its decision.

3. LLM-Powered Retention Strategy

The application integrates with the OpenRouter API to generate personalized retention recommendations. The LLM processes customer attributes and LIME insights to produce a meaningful and actionable strategy for preventing churn.

4. Streamlit Web Application

The application provides an intuitive web interface where users can input customer details, view predictions, explore LIME explanations, and read generated retention advice.

Project Structure
customer-main/
│── cu.py                         # Main Streamlit application
│── telco_churn_model.pkl         # Trained machine learning model
│── requirements.txt              # Python dependencies
│── Procfile                      # Deployment configuration
│── OPENROUTER_API_KEY.env        # Environment variable file (keep private)
│── CNAME                         # Custom domain configuration

Installation
1. Clone the Repository
git clone <your-repo-url>
cd customer-main

2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Add Your API Key

Create a .env file in the project directory:

OPENROUTER_API_KEY=your_api_key_here


This key enables the LLM recommendation feature.

Running the Application

Start the Streamlit server:

streamlit run cu.py


The application will open automatically in your browser, or you can visit:

http://localhost:8501

How the System Works
1. Customer Data Input

The user enters customer attributes such as contract type, payment method, monthly charges, tenure, and service usage.

2. Model Prediction

The system preprocesses the input and feeds it into the trained churn model to generate a prediction and a probability score.

3. LIME Explanation

LIME analyzes the prediction and generates a visual and textual explanation, identifying the most influential features.

4. LLM Recommendation

The application sends customer features and LIME-generated insights to the OpenRouter API.
The LLM generates tailored retention advice based on this combined context.

Deployment

The application includes a Procfile compatible with platforms such as:

Render

Railway

Heroku

Example Procfile entry:

web: streamlit run cu.py --server.port=$PORT


The CNAME file supports custom domain configuration.

Security Notice

The file OPENROUTER_API_KEY.env contains sensitive credentials.
This file should never be committed to version control or exposed publicly.

It is recommended to use environment variables on the hosting platform for production deployments.

License

MIT

Contact

For issues, improvements, or integration support, please contact the project owner which is moi
