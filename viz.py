import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import json
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import io
import time
import matplotlib.pyplot as plt
import seaborn as sns
import openai

# Accessing the API key from Streamlit's secrets
openai_api_key = st.secrets["openai"]["api_key"]

# LLM model configuration
llm = ChatOpenAI(
    model_name="gpt-4",
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

# Define categories for classification
categories = [
    "Food & Dining", "Utilities", "Transportation", "Entertainment & Recreation",
    "Shopping & Personal", "Healthcare", "Business Expenses", "Home & Rent",
    "Education", "Insurance", "Loan Payments", "Gifts & Donations", "Professional Services",
    "Taxes", "Miscellaneous/Other"
]

# Define the classification function using GPT
def classify_transaction_gpt(description):
    prompt = (
        f"Classify the following bank transaction description into one of these categories: "
        f"{', '.join(categories)}.\n"
        "Here are some examples to guide the classification:\n"
        "- 'Chevron', 'Exxon', 'Shell' -> 'Transportation'\n"
        "- 'Grocery', 'Food', 'Restaurant' -> 'Food & Dining'\n"
        "- 'Utilities', 'Electric', 'Gas Bill' -> 'Utilities'\n"
        "- 'Amazon', 'Walmart' -> 'Shopping & Personal'\n"
        "- Hotel or travel descriptions -> 'Entertainment & Recreation'\n\n"
        f"Now, classify this transaction:\n\nDescription: '{description}'"
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
        print(f"Error with description '{description}': {e}")
        return "Miscellaneous/Other"

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

# Streamlit interface setup
st.title("PDF Bank Statement Transaction Extractor and Analyzer")
st.write("Upload a PDF bank statement to extract transactions, classify, and visualize data.")

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if st.button("Process PDFs"):
    start_time = time.time()
    all_transactions = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Extract text from the PDF file
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
                    continue  # Ignore parts where JSON decoding fails

        if all_transactions:
            # Convert transaction data to a pandas DataFrame
            df_transactions = pd.DataFrame(all_transactions)

            # Display the transactions in the app
            st.write("Extracted Transactions")
            st.dataframe(df_transactions)

            # Clean and classify transactions
            df_transactions['amount'] = df_transactions['amount'].replace({'\$': '', ',': ''}, regex=True).astype(float)
            df_transactions['category'] = df_transactions['description'].apply(classify_transaction_gpt)

            # Display classified transactions
            st.write("Classified Transactions")
            st.dataframe(df_transactions[['trans_date', 'description', 'amount', 'category']])

            # Visualization
            st.write("Transaction Category Distribution")
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df_transactions, x='category')
            plt.xticks(rotation=45)
            st.pyplot(plt)

            # Download option
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_transactions.to_excel(writer, index=False)
                buffer.seek(0)
            st.download_button(
                label="Download transactions as Excel",
                data=buffer,
                file_name="transactions_with_categories.xlsx"
            )

        else:
            st.warning("No transactions were identified.")
    else:
        st.warning("Please upload at least one PDF file.")
    
    end_time = time.time()
    st.write(f"Processing completed in {end_time - start_time:.2f} seconds.")
