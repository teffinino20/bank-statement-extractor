import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import json
import io
import time
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

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
Extract the following information from the provided bank statement text in a strict JSON format:
Transaction Date, Description, Amount (include sign if it's negative or a debit transaction), Currency (if mentioned), and Type of transaction (Debit or Credit).

Avoid information related to: "Previous Balance", "Payments/Credits", "New Charges", 
        "Fees", "Interest Charged", "Balance", "Total", "Opening balance", 
        "Closing balance", "Account Summary", "Account Activity Details", 
        "Minimum Due", "Available and Pending", "Closing Date", "Payment Due Date",
        "Due Date"
        
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

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Function to clean the text
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
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

# Streamlit interface configuration
st.title("PDF Bank Statement Transaction Extractor")
st.write("Upload a PDF bank statement to extract transactions.")

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if st.button("Process PDFs"):
    all_transactions = []
    start_time = time.time()

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Extract text from the PDF file
            extracted_text = extract_text_from_pdf(uploaded_file)
            cleaned_text = clean_text(extracted_text)
            text_parts = split_text(cleaned_text)

            # Extract transactions using LLM
            for idx, part in enumerate(text_parts):
                try:
                    transactions_data = transaction_chain.predict(text=part)
                    parsed_transactions = json.loads(transactions_data)
                    if isinstance(parsed_transactions, list):
                        all_transactions.extend(parsed_transactions)
                except json.JSONDecodeError:
                    st.error(f"Error decoding JSON for part {idx + 1} of {uploaded_file.name}")
                except openai.error.OpenAIError as e:
                    st.error(f"OpenAI API error: {str(e)}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")

        if all_transactions:
            # Convert the transaction data into a pandas DataFrame
            df_transactions = pd.DataFrame(all_transactions)

            # Display the transactions in the app
            st.write("Extracted Transactions")
            st.dataframe(df_transactions)

            # Option to download the Excel file
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_transactions.to_excel(writer, index=False)

            st.download_button(
                label="Download transactions as Excel",
                data=buffer.getvalue(),
                file_name="transactions.xlsx"
            )
        else:
            st.warning("No transactions were identified.")
    else:
        st.warning("Please upload at least one PDF file.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    st.write(f"Processing time: {elapsed_time:.2f} seconds")
