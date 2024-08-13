import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import json
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Accessing the API key from Streamlit's secrets
openai_api_key = st.secrets["openai"]["api_key"]

# Function to extract text from PDFs
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
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

# Function to split the text into smaller parts
def split_text(text, max_length=3000):
    """Split the text into smaller parts with a specified maximum length."""
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# LLM Configuration
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
Extract the following information from the provided bank statement text in a strict JSON format:
Transaction Date, Description, Amount (include sign if it's negative or a debit transaction), Currency (if mentioned), and Type of transaction (Debit or Credit).

Ensure the JSON is properly formatted with no additional text.

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

# Streamlit UI Configuration
st.title("PDF Bank Statement Transaction Extractor")
st.write("Upload a PDF bank statement to extract transactions.")

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if st.button("Process PDFs"):
    all_transactions = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Extract text from the uploaded PDF
            extracted_text = extract_text_from_pdf(uploaded_file)
            cleaned_text = clean_text(extracted_text)
            processed_texts = split_text(cleaned_text)

            # Extract transactions using LLM
            for part_idx, text in enumerate(processed_texts):
                transactions_data = transaction_chain.predict(text=text)
                st.write(f"Processing part {part_idx + 1}...")

                # Print raw output for debugging
                st.write(f"Raw output for part {part_idx + 1}:\n{transactions_data}\n")
                
                try:
                    parsed_transactions = json.loads(transactions_data)
                    if isinstance(parsed_transactions, list):
                        all_transactions.extend(parsed_transactions)
                    else:
                        st.error(f"Output is not a valid list for part {part_idx + 1}.")
                except json.JSONDecodeError as e:
                    st.error(f"Error decoding JSON for part {part_idx + 1} of {uploaded_file.name}: {e}")
                    st.error(f"Raw output: {transactions_data}")

        if all_transactions:
            # Convert the transaction data into a pandas DataFrame
            df_transactions = pd.DataFrame(all_transactions)

            # Display the transactions in the app
            st.write("Extracted Transactions")
            st.dataframe(df_transactions)

            # Option to download the Excel file
            st.download_button(
                label="Download transactions as Excel",
                data=df_transactions.to_excel(index=False),
                file_name="transactions.xlsx"
            )
        else:
            st.warning("No transactions were identified.")
    else:
        st.warning("Please upload at least one PDF file.")
