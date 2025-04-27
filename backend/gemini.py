import os
import google.generativeai as genai
import dotenv

dotenv.load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.0-flash")

def generate_response(new_review: str, embedding: list):
    prompt = f"""
You are a polite restaurant assistant.

Customer review: "{new_review}"
Reply to the customer professionally and courteously.
"""
    res = model.generate_content(prompt)
    return res.text.strip()

def generate_summary(reviews: list, embeddings: list):
    joined_reviews = "\n".join(f"- {rev}" for rev in reviews)
    print(joined_reviews)
    prompt = f"""
Summarize the following customer reviews into 3-4 lines highlighting the key sentiments:

{joined_reviews}
"""
    res = model.generate_content(prompt)
    return res.text.strip()