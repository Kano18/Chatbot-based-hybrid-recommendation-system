# app.py
import os
import csv
import random
from datetime import datetime, date

import streamlit as st
import pandas as pd

# --- OpenAI client ---
from openai import OpenAI
OPENAI_API_KEY = os.getenv("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", "")
client = OpenAI(api_key="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")

# --- Your project helpers (unchanged import) ---
from llmchatbot import (
    create_personalized_prompt, 
    get_openai_context, 
    format_user_info, 
    second_llm_prompt, 
    get_openai_context2, 
    detect_seasonality, 
    get_seasonal_user_recommendations_with_names
)

# --- AWS (S3) setup: optional but used if creds are present ---
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError

AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")  
S3_BUCKET = os.getenv("S3_BUCKET", "fashion-ai-chatbot-bucket")  

def get_s3_client():
    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        # a cheap health check call (will raise if no creds)
        s3.list_buckets()
        return s3
    except Exception:
        return None

S3 = get_s3_client()

# ---------- Utility: read CSV from S3 with local fallback ----------
def read_csv_s3_or_local(key: str, local_path: str) -> pd.DataFrame:
    """
    Tries S3: s3://S3_BUCKET/key
    Falls back to local file if S3 unavailable or key not found.
    """
    if S3:
        try:
            obj = S3.get_object(Bucket=S3_BUCKET, Key=key)
            return pd.read_csv(obj["Body"])
        except (ClientError, BotoCoreError, NoCredentialsError, FileNotFoundError):
            pass
    # local fallback
    return pd.read_csv(local_path)

# ---------- Utility: write a small text payload to S3 (or local) ----------
def write_text_to_s3_or_local(text: str, key: str, local_path: str):
    if S3:
        try:
            S3.put_object(Bucket=S3_BUCKET, Key=key, Body=text.encode("utf-8"))
            return
        except (ClientError, BotoCoreError, NoCredentialsError):
            pass
    # local fallback (append)
    with open(local_path, "a", encoding="utf-8") as f:
        f.write(text)

# ---------- Data loaders (cached) ----------
@st.cache_data
def load_user_data() -> pd.DataFrame:
    return read_csv_s3_or_local(
        key="processed_customer_final.csv",
        local_path="processed_customer_final.csv",
    )

@st.cache_data
def load_product_data():
    df = read_csv_s3_or_local(
        key="processed_product_final.csv",
        local_path="processed_product_final.csv",
    )
    product_list = (
        df[["productDisplayName", "id"]]
        .dropna()
        .drop_duplicates(subset="id")
        .to_dict(orient="records")
    )
    return df, product_list

@st.cache_data
def load_seasonal_recommendations():
    return {
        "spring": read_csv_s3_or_local("Spring_user_recommendations.csv", "Spring_user_recommendations.csv"),
        "summer": read_csv_s3_or_local("Summer_user_recommendations.csv", "Summer_user_recommendations.csv"),
        "fall":   read_csv_s3_or_local("Fall_user_recommendations.csv",   "Fall_user_recommendations.csv"),
        "winter": read_csv_s3_or_local("Winter_user_recommendations.csv", "Winter_user_recommendations.csv"),
    }

def calculate_age(birthdate):
    today = date.today()
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))

# ---------- Persist users: write-through to S3 (CSV) with local fallback ----------
def persist_users_df(df: pd.DataFrame):
    # Write locally first (safe baseline)
    df.to_csv("processed_customer_final.csv", index=False)
    # Then mirror to S3 (optional)
    if S3:
        try:
            S3.upload_file("processed_customer_final.csv", S3_BUCKET, "processed_customer_final.csv")
        except (ClientError, BotoCoreError, NoCredentialsError):
            pass

# ---------- Simple auth helpers ----------
def login(username, email, users_df):
    return bool(
        (users_df["username"] == username).any() and
        (users_df["email"] == email).any()
    )

def signup(first_name, last_name, username, email, gender, birthdate, home_location, home_country, past_trans):
    users = load_user_data().copy()
    # new id
    customer_id = 1 if users.empty else int(users["customer_id"].max()) + 1
    age = calculate_age(birthdate)
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
        "past_trans": past_trans,
    }
    users = pd.concat([users, pd.DataFrame([new_user])], ignore_index=True)
    persist_users_df(users)
    # clear cache so new user appears immediately
    load_user_data.clear()
    return "Sign up successful"

# ---------- Logging ----------
def save_chat_history(user_query: str, chatbot_response: str):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    payload = f"[{ts}] User: {user_query}\n[{ts}] Chatbot: {chatbot_response}\n\n"
    # store each interaction as its own S3 object (no overwrite issues)
    key = f"logs/chat_{ts}.txt"
    write_text_to_s3_or_local(payload, key, "chat_history.txt")

# ---------- Streamlit UI ----------
def main():
    st.title("Fashion E-commerce Chatbot")
    menu = ["Login", "Sign Up"]
    choice = st.sidebar.selectbox("Menu", menu)

    users_df = load_user_data()
    product_df, product_list = load_product_data()
    seasonal = load_seasonal_recommendations()

    if choice == "Sign Up":
        with st.form("Signup Form", clear_on_submit=True):
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            username = st.text_input("Username")
            email = st.text_input("Email")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            birthdate = st.date_input("Birthdate", min_value=date(1950, 1, 1), max_value=date.today())
            home_location = st.text_input("Home Location")
            home_country = st.text_input("Home Country")
            past_trans = "[]"
            submit_button = st.form_submit_button("Sign Up")

            if submit_button:
                if not (first_name and last_name and username and email):
                    st.error("Please complete all required fields.")
                else:
                    result = signup(first_name, last_name, username, email, gender, birthdate, home_location, home_country, past_trans)
                    st.success(result)

    elif choice == "Login":
        with st.form("Login Form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            submit_button = st.form_submit_button("Login")
            if submit_button:
                if login(username, email, users_df):
                    user = users_df[(users_df['username'] == username) & (users_df['email'] == email)].iloc[0]
                    st.session_state['user'] = user.to_dict()
                    st.session_state['successful_login'] = True
                    st.success("Login successful. Redirecting to chatbot...")
                else:
                    st.error("Invalid username or email")

    # Chat area
    if st.session_state.get("successful_login"):
        st.write(f"Welcome, {st.session_state['user']['first_name']}! You can now interact with our chatbot.")
        user_query = st.text_input("Ask something about our fashion line:")

        if st.button("Get Recommendations"):
            user_id = st.session_state['user']['customer_id']

            # season from query
            season = detect_seasonality(user_query)

            # recs from your helper
            recommendations = get_seasonal_user_recommendations_with_names(season, user_id, product_df)

            # format user context
            user_info = format_user_info(user_id, users_df)

            # build LLM context (your existing helper)
            context = get_openai_context2(
                second_llm_prompt,
                chat_history="",
                user_info=user_info,
                user_query=user_query,
                season=season,
                first_llm_suggestions="Initial suggestions if any",
                product_list=product_list
            )

            # present
            st.subheader("Chatbot Response")
            st.write(context)

            # optional: show recommendations list/table if available
            if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
                st.subheader("Recommended Products")
                st.dataframe(recommendations)
            elif isinstance(recommendations, list) and recommendations:
                st.subheader("Recommended Products")
                st.write(recommendations)

            # log
            save_chat_history(user_query, context)

    else:
        st.info("Please log in to access the chatbot.")

if __name__ == "__main__":
    # Safety: warn if OPENAI_API_KEY missing
    if not OPENAI_API_KEY:
        st.warning("OPENAI_API_KEY is not set. Set it in your environment for full functionality.")
    main()



#### to run on Terminal
# sudo apt update && sudo apt install python3-pip -y
#pip install -r requirements.txt
#streamlit run app2.py --server.port 8501
