# Import necessary libraries
import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import pandas as pd
import json
import openai
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import matplotlib.dates as mdates
import time
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Accessing the API key from Streamlit's secrets
openai_api_key = st.secrets["openai"]["api_key"]

# Set up OpenAI API key for classification
openai.api_key = openai_api_key

# LLM model configuration for extraction
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    max_tokens=4096,
    openai_api_key=openai_api_key
)

# Define the response schemas for the parser
response_schemas = [
    ResponseSchema(name="trans_date", description="Transaction date"),
    ResponseSchema(name="description", description="The transaction description"),
    ResponseSchema(name="amount", description="Amount of the transaction"),
    ResponseSchema(name="currency", description="Currency of the transaction", optional=True),
    ResponseSchema(name="type of transaction", description="Type of the transaction: Debit or Credit", optional=True)
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Define the prompt template for transaction extraction
prompt_template = """
Extract the following information from the provided bank statement text in JSON format:
Transaction Date, Description, Amount (include sign if it's negative or a debit transaction), Currency (if mentioned), and Type of transaction (Debit or Credit).

Avoid information related to: "Previous Balance", "Payments/Credits", "New Charges", 
        "Fees", "Interest Charged", "Balance", "Total", "Opening balance", 
        "Closing balance", "Account Summary", "Account Activity Details", 
        "Minimum Due", "Available and Pending", "Closing Date", "Payment Due Date",
        "Due Date"

Bank Statement:
{text}

Extracted Transactions:
"""

transaction_prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"],
    output_parser=output_parser
)

# Create the LLM chain for transaction extraction
transaction_chain = LLMChain(
    llm=llm,
    prompt=transaction_prompt
)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Function to clean the text by removing non-relevant information
def clean_text(text):
    """Clean the text by removing non-relevant information."""
    lines = text.split('\n')
    cleaned_lines = []
    summary_keywords = [
        "Previous Balance", "Payments/Credits", "New Charges", 
        "Fees", "Interest Charged", "Balance", "Total", "Opening balance", 
        "Closing balance", "Account Summary", "Account Activity Details", 
        "Minimum Due", "Available and Pending", "Closing Date", "Payment Due Date",
        "Due Date"
    ]
    
    for line in lines:
        if any(keyword in line for keyword in summary_keywords):
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

# Function to split text into smaller parts
def split_text(text, max_length=3000):
    """Split the text into smaller parts with a specified maximum length."""
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# Define the classification categories
categories = [
    "Food & Dining", "Utilities", "Transportation", "Entertainment & Recreation",
    "Shopping & Personal", "Healthcare", "Business Expenses", "Home & Rent",
    "Education", "Insurance", "Loan Payments", "Gifts & Donations", "Professional Services",
    "Taxes", "Miscellaneous/Other"
]

# Define the GPT-powered classification function
def classify_transaction_gpt(description):
    prompt = (
        f"Classify the following bank transaction description into one of these categories: "
        f"{', '.join(categories)}. If the description doesn't fit any of these categories, "
        f"respond with 'Miscellaneous/Other'.\n\nDescription: '{description}'"
    )
    try:
        # Using ChatCompletion for OpenAI API compatibility
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        category = response['choices'][0]['message']['content'].strip()
        return category if category in categories else "Miscellaneous/Other"
    except Exception as e:
        print(f"Error with description '{description}': {e}")
        return "Miscellaneous/Other"

# Streamlit interface setup
st.title("PDF Bank Statement Transaction Extractor and Analyzer")
st.write("Upload a PDF bank statement to extract and analyze transactions.")

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if st.button("Process PDFs"):
    start_time = time.time()
    all_transactions = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            extracted_text = extract_text_from_pdf(uploaded_file)
            cleaned_text = clean_text(extracted_text)
            processed_texts = split_text(cleaned_text)

            # Extract transactions using LLM
            for text in processed_texts:
                try:
                    transactions_data = transaction_chain.predict(text=text)
                    parsed_transactions = json.loads(transactions_data)
                    if isinstance(parsed_transactions, list):
                        all_transactions.extend(parsed_transactions)
                except json.JSONDecodeError:
                    continue

        if all_transactions:
            # Convert transaction data to DataFrame and process
            df = pd.DataFrame(all_transactions)

            # Classification and amount cleaning
            df['Category'] = df['description'].apply(classify_transaction_gpt)
            df['Amount'] = df['amount'].replace({'\$': '', ',': ''}, regex=True).astype(float)
            df = df[~df['description'].isin(["Payment", "ONLINE PAYMENT - THANK YOU"])]

            # Visualizations
            st.write("Extracted and Classified Transactions")
            st.dataframe(df[['trans_date', 'description', 'Amount', 'Category']])

            # Visualization 1: Transaction count by category
            plt.figure(figsize=(10, 6))
            sns.countplot(x='Category', data=df)
            plt.title("Transaction Count by Category")
            plt.xticks(rotation=45)
            st.pyplot(plt)

            # Visualization 2: Pie chart for transaction count by category
            category_counts = df['Category'].value_counts()
            plt.figure(figsize=(8, 8))
            category_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
            plt.title("Transaction Count by Category")
            st.pyplot(plt)

            # Running Balance Plot
            df['trans_date'] = pd.to_datetime(df['trans_date'])
            df['Running Balance'] = df['Amount'].cumsum()
            plt.figure(figsize=(12, 6))
            plt.plot(df['trans_date'], df['Running Balance'], marker='o')
            plt.title("Running Balance Over Time")
            plt.xticks(rotation=45)
            st.pyplot(plt)

            # Scatter Plot by Date and Amount
            plt.figure(figsize=(12, 6))
            sns.scatterplot(data=df, x='trans_date', y='Amount', hue='Category', palette="Set2", s=100, alpha=0.7)
            plt.title("Scatter Plot of Transactions by Date and Amount")
            plt.xticks(rotation=45)
            plt.grid(True)
            st.pyplot(plt)

            # Monthly Spending Forecast
            df['Month'] = df['trans_date'].dt.to_period("M")
            monthly_spending = df.groupby('Month')['Amount'].sum()
            plt.figure(figsize=(10, 5))
            monthly_spending.plot(kind='line', label="Actual Spending")
            monthly_spending.rolling(window=3).mean().plot(label="3-Month Rolling Avg", linestyle='--')
            plt.title("Monthly Spending Forecast")
            plt.xlabel("Month")
            plt.legend()
            st.pyplot(plt)

        else:
            st.warning("No transactions were identified.")
    else:
        st.warning("Please upload at least one PDF file.")
    
    end_time = time.time()
    st.write(f"Processing completed in {end_time - start_time:.2f} seconds.")

