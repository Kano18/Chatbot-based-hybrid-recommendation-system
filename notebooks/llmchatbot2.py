import openai
from openai import OpenAI
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from scipy.spatial.distance import cdist

# Initialize OpenAI Client
#client = OpenAI(api_key="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")

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("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"))


# Load data files
product = pd.read_csv('processed_product_final.csv').iloc[:, 1:]
customer_df = pd.read_csv('processed_customer_final.csv')
Spring_user_recommendations = pd.read_csv('Spring_user_recommendations.csv')
Summer_user_recommendations = pd.read_csv('Summer_user_recommendations.csv')
Fall_user_recommendations = pd.read_csv('Fall_user_recommendations.csv')
Winter_user_recommendations = pd.read_csv('Winter_user_recommendations.csv')

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L12-v2")

# Function to create a personalized prompt
def create_personalized_prompt(customer_df, customer_id):
    """Create a personalized system prompt with user-specific details."""
    user_info = customer_df[customer_df['customer_id'] == customer_id]
    if not user_info.empty:
        user_details = f'''
        Current user details:
        - Gender: {user_info.iloc[0]['gender']}
        - Age: {user_info.iloc[0]['age']}
        - Location: {user_info.iloc[0]['home_location']}
        - Country: {user_info.iloc[0]['home_country']}
        - Past Transactions: {", ".join(user_info.iloc[0]['past_trans'])}
        '''
    else:
        user_details = "No user details are available."

    personalized_prompt = f'''
        You are an apparel recommender agent for an Indonesian apparel company. {user_details}
        Your job is to suggest different types of apparel one can wear based on the user's query and user details provided. You can understand the occasion and recommend the correct apparel items for the occasion if applicable, or just output that specific apparels if user is already very specific. Below are a few examples with reasons as to why the particular item is recommended:

        User question - show me blue shirts
        Your response - blue shirts
        Reason for recommendation - user is already specific in their query, nothing to recommend

        User question - What can I wear for office party?
        Your response - semi-formal dress, suit, office party, dress
        Reason for recommendation - recommend apparel choices based on the occasion

        User question - I am doing shopping for trekking in mountains; what do you suggest?
        Your response - heavy jacket, jeans, boots, windshield, sweater.
        Reason for recommendation - recommend apparel choices based on the occasion
    '''
    return personalized_prompt

# Function to get response from OpenAI's GPT-3.5 Turbo model
def get_openai_context(prompt: str, user_query: str) -> str:
    """Get responses from OpenAI model."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=1,
    )
    # Use response.model_dump() if you need a dictionary structure
    return response.choices[0].message.content

# Detect seasonality based on user query or current date
def detect_seasonality(user_query):
    seasons = {
        "winter": ["winter", "cold", "snow", "chilly", "freezing"],
        "spring": ["spring", "bloom", "flowers", "mild", "rain"],
        "summer": ["summer", "hot", "sunny", "warm"],
        "fall": ["fall", "autumn", "leaves", "cool", "crisp"]
    }
    for season, keywords in seasons.items():
        if any(keyword in user_query.lower() for keyword in keywords):
            return season
    
    month = datetime.now().month
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    elif month in [9, 10, 11]:
        return "fall"

# Function to get seasonal user recommendations with product names
def get_seasonal_user_recommendations_with_names(season, user_id, product_df):
    if season == "spring":
        recommendations_df = Spring_user_recommendations
    elif season == "summer":
        recommendations_df = Summer_user_recommendations
    elif season == "fall":
        recommendations_df = Fall_user_recommendations
    elif season == "winter":
        recommendations_df = Winter_user_recommendations
    else:
        return "No recommendations available for this season."

    user_recommendations = recommendations_df[recommendations_df['customer_id'] == user_id]
    if not user_recommendations.empty:
        user_recommendations_with_names = user_recommendations.merge(
            product_df[['id', 'productDisplayName']],
            left_on='product_id',
            right_on='id',
            how='left'
        ).drop(columns=['id'])
        return user_recommendations_with_names
    else:
        return f"No recommendations found for user ID {user_id} in {season}."

# Function to fetch OpenAI context
def get_openai_context1(prompt:str, chat_history:str) -> str:
    """Get responses from OpenAI model."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": chat_history}
        ],
        temperature=1,
    )
    return response.choices[0].message.content

def format_user_info(user_id, df):
    # Fetch user data based on the user_id from the DataFrame
    user_data = df[df['customer_id'] == user_id]
    if not user_data.empty:
        user_data = user_data.iloc[0]
        # Create a formatted string of user details
        user_info = (
            f'''Current user details:
            - Gender: {user_data['gender']}
            - Age: {user_data['age']}
            - Location: {user_data['home_location']}
            - Country: {user_data['home_country']}
            - Past Transactions: {", ".join(user_data['past_trans']) if user_data['past_trans'] else 'None'}
            ''')
        return user_info
    else:
        return "No user details are available."

second_llm_prompt = (
    """
    You can recommendation engine chatbot agent for an Indonesian apparel brand.
    You are provided with users questions and some apparel recommendations from the brand's database. Provide the Name and Product ID.
    If there are duplicate product names, only show the first match.
    Your job is to present the most relevant items from the data give to you.
    Do not answer anything else apart from apparel recommendation from the company's database.
    Do not suggest anything else that is not relevant to the user's query.
    Do not suggest colors that don't match the user's query.
    """
)

# Function to fetch OpenAI context with additional context
def get_openai_context2(prompt, chat_history, user_info, user_query, season, first_llm_suggestions, recommendations):
    recommendations_text = (
        recommendations.to_string(index=False) if isinstance(recommendations, pd.DataFrame) else recommendations
    )
    enhanced_prompt = (
        f"{prompt}\n"
        f"Here is what we know about the user:\n{user_info}\n\n"
        f"User Query: {user_query}\n"
        f"Detected Season: {season.capitalize()}\n"
        f"Apparel Suggestions from Initial Model:\n{first_llm_suggestions}\n\n"
        f"Final Recommendations from Database:\n{recommendations_text}\n\n"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": enhanced_prompt},
            {"role": "user", "content": chat_history}
        ],
        temperature=1,
    )
    return response.choices[0].message.content
