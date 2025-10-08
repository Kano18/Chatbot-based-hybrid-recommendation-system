from openai import OpenAI
import csv
from datetime import datetime, date
import streamlit as st
import pandas as pd
import openai
import random
from llmchatbot import (
    create_personalized_prompt, 
    get_openai_context, 
    format_user_info, 
    second_llm_prompt, 
    get_openai_context2, 
    detect_seasonality, 
    get_seasonal_user_recommendations_with_names
)

# Set OpenAI API key
#client = OpenAI(api_key="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")
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("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"))

# Load user and product data
@st.cache_data 
def load_user_data():
    return pd.read_csv('processed_customer_final.csv')

@st.cache_data
def load_product_data():
    # Load the full product DataFrame from the CSV
    product_df = pd.read_csv('processed_product_final.csv')
    # Extract productDisplayName and productID for the product list
    product_list = product_df[['productDisplayName', 'id']].dropna().drop_duplicates(subset='id').to_dict(orient='records')
    return product_df, product_list

# Load seasonal recommendation data
@st.cache_data
def load_seasonal_recommendations():
    return {
        "spring": pd.read_csv('Spring_user_recommendations.csv'),
        "summer": pd.read_csv('Summer_user_recommendations.csv'),
        "fall": pd.read_csv('Fall_user_recommendations.csv'),
        "winter": pd.read_csv('Winter_user_recommendations.csv')
    }

def calculate_age(birthdate):
    today = date.today()
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))

# Load data
users_df = load_user_data()
product_df, product_list = load_product_data() 

# Log activity
def log_activity(action, user_id):
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
    with open('user_activity.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, action, user_id])

# Login function
def login(username, email):
    if username in users_df["username"].values and email in users_df["email"].values:
        return True
    else:
        return False

# Signup function
def signup(first_name, last_name, username, email, gender, birthdate, home_location, home_country, past_trans):
    users_df = load_user_data()
    
    # Generate customer_id
    if users_df.empty:
        customer_id = 1
    else:
        customer_id = users_df["customer_id"].max() + 1
    
    age = calculate_age(birthdate)

    # Add new user to DataFrame
    new_user = {
        "customer_id": customer_id,
        "first_name": first_name,
        "last_name": last_name,
        "username": username,
        "email": email,
        "gender": gender,
        "birthdate": birthdate,
        "age": age,
        "home_location": home_location,
        "home_country": home_country,
        "past_trans": past_trans
    }
    users_df = users_df._append(new_user, ignore_index=True)
    users_df.to_csv("processed_customer_final.csv", index=False)
    
    return "Sign up successful"

# Save chat history
def save_chat_history(user_query, chatbot_response):
    with open("chat_history.txt", "a") as file:
        file.write(f"User: {user_query}\n")
        file.write(f"Chatbot: {chatbot_response}\n\n")

# Streamlit UI
def main():
    st.title("Fashion E-commerce Chatbot")
    menu = ["Login", "Sign Up"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Sign Up":
        with st.form("Signup Form"):
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            username = st.text_input("Username")
            email = st.text_input("Email")
            gender = st.selectbox("Gender", ["Male", "Female"])
            birthdate = st.date_input("Birthdate", min_value=date(1950, 1, 1), max_value=date.today())
            home_location = st.text_input("Home Location")
            home_country = st.text_input("Home Country")
            past_trans = "[]"
            submit_button = st.form_submit_button("Sign Up")

            if submit_button:
                result = signup(first_name, last_name, username, email, gender, birthdate, home_location, home_country, past_trans)
                st.success(result)

    elif choice == "Login":
        with st.form("Login Form"):
            users_df = load_user_data()
            username = st.text_input("Username")
            email = st.text_input("Email")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if login(username, email):
                    users_df = load_user_data()
                    user = users_df[(users_df['username'] == username) & (users_df['email'] == email)].iloc[0]
                    st.session_state['user'] = user.to_dict()
                    st.success("Login successful. Redirecting to chatbot...")
                    st.session_state['successful_login'] = True
                else:
                    st.error("Invalid username or email")

    # Check if the user has logged in
    if st.session_state.get("successful_login"):
        st.write(f"Welcome, {st.session_state['user']['first_name']}! You can now interact with our chatbot.")
        user_query = st.text_input("Ask something about our fashion line:")

        if st.button("Get Recommendations"):
            # Use the session user ID instead of a hardcoded user_id
            user_id = st.session_state['user']['customer_id']
            
            # Detect season based on user query
            season = detect_seasonality(user_query)
            
            # Get recommendations based on user ID and detected season
            recommendations = get_seasonal_user_recommendations_with_names(season, user_id, product_df)

            # Format user info
            user_info = format_user_info(user_id, users_df)

            # Generate context with enhanced user details and actual query, including the product list
            context = get_openai_context2(
                second_llm_prompt,
                chat_history="",
                user_info=user_info,
                user_query=user_query,
                season=season,
                first_llm_suggestions="Initial suggestions if any",
                product_list=product_list  # Pass the product list here
            )
            # Display the chatbot response
            st.write(context)

            # Save chat history
            save_chat_history(user_query, context)

    else:
        st.error("Please log in to access the chatbot.")

if __name__ == "__main__":
    main()
