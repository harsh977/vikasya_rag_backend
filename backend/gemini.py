import os
import google.generativeai as genai
import dotenv

dotenv.load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.0-flash")

# Function to generate polite responses to customer reviews
def generate_response(new_review: str, embedding: list):
    prompt = f"""
You are a polite and professional public service assistant.

Customer review: "{new_review}"
Reply to the customer courteously, acknowledging their feedback, and providing any relevant guidance or steps for improvement.
"""
    res = model.generate_content(prompt)
    return res.text.strip()

# Function to generate a summary of multiple public service reviews
def generate_summary(reviews: list, embeddings: list):
    # Combine all the reviews into a single paragraph for summary generation
    joined_reviews = "\n".join(f"- {rev}" for rev in reviews)
    print(joined_reviews)
    prompt = f"""
Summarize the following customer reviews regarding public services into 5-6 lines , highlighting the key sentiments, areas for improvement, and any recurring themes , display point wise for better understanding.
kust summarize the reviews , dont ask to provide any feedback or response to the reviews.:

{joined_reviews}
"""
    res = model.generate_content(prompt)
    return res.text.strip()
