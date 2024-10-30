# Import necessary libraries
import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import pandas as pd
import json
import openai
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import time

# Set up OpenAI API key
openai_api_key = st.secrets["openai"]["api_key"]
openai.api_key = openai_api_key

# Define categories for classification
categories = [
    "Food & Dining", "Utilities", "Transportation", "Shopping & Personal",
    "Healthcare", "Business Expenses", "Home & Rent", "Education", "Insurance",
    "Loan Payments", "Gifts & Donations", "Professional Services", "Taxes", 
    "Miscellaneous/Other"
]

# Configure the LLM model
llm = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=4096, openai_api_key=openai_api_key)

# Define the response schemas and the parser
response_schemas = [
    ResponseSchema(name="Trans_Date", description="Transaction date"),
    ResponseSchema(name="Description", description="Transaction description"),
    ResponseSchema(name="Amount", description="Transaction amount"),
    ResponseSchema(name="Currency", description="Currency of the transaction", optional=True),
    ResponseSchema(name="Type_of_Transaction", description="Type of transaction: Debit or Credit", optional=True)
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Define prompt template for transaction extraction
prompt_template = """
Extract the following information in JSON format:
Transaction Date, Description, Amount (include sign for debit transactions), Currency (if mentioned), and Type of transaction (Debit or Credit).
Avoid details like 'Previous Balance', 'Payments/Credits', 'New Charges', 'Interest Charged'.

Bank Statement:
{text}

Extracted Transactions:
"""
transaction_prompt = PromptTemplate(template=prompt_template, input_variables=["text"], output_parser=output_parser)
transaction_chain = LLMChain(llm=llm, prompt=transaction_prompt)

# Functions for extraction and processing
def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

def classify_transaction_gpt(description):
    """Classify transaction description using GPT-based prompt."""
    prompt = (
        f"Classify this bank transaction into one of these categories: {', '.join(categories)}.\n"
        "Here are some examples:\n"
        "- 'Chevron', 'Shell': Transportation\n"
        "- 'Grocery', 'Food', 'Restaurant': Food & Dining\n"
        "- 'Utilities', 'Electric', 'Gas': Utilities\n"
        "- 'Amazon', 'Walmart': Shopping & Personal\n"
        "- Hotel or travel-related: Entertainment & Recreation\n\n"
        f"Description: '{description}'"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        category = response['choices'][0]['message']['content'].strip()
        return category if category in categories else "Miscellaneous/Other"
    except Exception as e:
        st.warning(f"Classification error for '{description}': {e}")
        return "Miscellaneous/Other"

# Streamlit interface setup
st.title("Enhanced Bank Statement Analysis")
st.write("Upload a PDF bank statement to extract and analyze transactions.")

# File upload and processing
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
if st.button("Process PDFs"):
    start_time = time.time()
    all_transactions = []

    if uploaded_files:
        for pdf_file in uploaded_files:
            extracted_text = extract_text_from_pdf(pdf_file)
            parts = [extracted_text[i:i+3000] for i in range(0, len(extracted_text), 3000)]
            
            for part in parts:
                try:
                    transactions_data = transaction_chain.predict(text=part)
                    parsed_transactions = json.loads(transactions_data)
                    all_transactions.extend(parsed_transactions)
                except json.JSONDecodeError:
                    continue

        # Create DataFrame from transactions
        if all_transactions:
            df = pd.DataFrame(all_transactions)
            
            # Verify if 'Description' column exists
            if 'Description' not in df.columns:
                st.error("The 'Description' column is missing in the extracted data. Check the extraction prompt and JSON parsing.")
            else:
                # Classification and cleaning
                df['Category'] = df['Description'].apply(classify_transaction_gpt)
                df['Amount'] = pd.to_numeric(df['Amount'].replace({'\$': '', ',': ''}, regex=True))
                df = df[~df['Description'].isin(["Payment", "ONLINE PAYMENT - THANK YOU"])]

                # Visualizations
                st.write("### Transaction Summary Table")
                st.dataframe(df[['Trans_Date', 'Description', 'Amount', 'Category']])

                # Visualization 1: Transaction count by category
                plt.figure(figsize=(10, 6))
                sns.countplot(x='Category', data=df)
                plt.title("Transaction Count by Category")
                plt.xticks(rotation=45)
                st.pyplot(plt)

                # Visualization 2: Spending by category (pie chart)
                plt.figure(figsize=(8, 8))
                df['Category'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140)
                plt.title("Transaction Distribution by Category")
                st.pyplot(plt)

                # Visualization 3: Running Balance Plot
                df['Running Balance'] = df['Amount'].cumsum()
                plt.figure(figsize=(12, 6))
                plt.plot(df['Trans_Date'], df['Running Balance'])
                plt.title("Running Balance Over Time")
                st.pyplot(plt)

                # Download button for processed data
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False)
                st.download_button(
                    label="Download transactions as Excel",
                    data=buffer,
                    file_name="processed_transactions.xlsx"
                )
        else:
            st.warning("No transactions were identified.")
    else:
        st.warning("Please upload at least one PDF file.")
    st.write(f"Processing completed in {time.time() - start_time:.2f} seconds.")
