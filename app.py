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

# Accessing the API key from Streamlit's secrets
openai_api_key = st.secrets["openai"]["api_key"]

# LLM model configuration
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    max_tokens=4096,
    openai_api_key=openai_api_key
)

# Define response schemas for the parser
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

# Streamlit interface setup
st.title("Enhanced PDF Bank Statement Transaction Extractor")
st.write("Upload a PDF bank statement to extract and visualize transactions.")

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# Define expense categories
expense_categories = {
    "Restaurants": ["restaurant", "cafe", "dining"],
    "Utilities": ["electricity", "water", "gas", "internet"],
    "Entertainment": ["cinema", "theater", "concert", "movie"],
    "E-commerce": ["amazon", "shop", "online"],
    # Add more categories as needed
}

def classify_transaction(description):
    for category, keywords in expense_categories.items():
        if any(keyword.lower() in description.lower() for keyword in keywords):
            return category
    return "Other"

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

            # Classify transactions
            df_transactions['Category'] = df_transactions['description'].apply(classify_transaction)

            # Display the transactions in the app
            st.write("Extracted and Classified Transactions")
            st.dataframe(df_transactions)

            # Option to download the Excel file
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_transactions.to_excel(writer, index=False)
                buffer.seek(0)

            st.download_button(
                label="Download transactions as Excel",
                data=buffer,
                file_name="classified_transactions.xlsx"
            )

            # Visualization options
            st.subheader("Visualizations")
            
            # Transaction count by category
            st.write("Transaction count by category:")
            category_counts = df_transactions['Category'].value_counts()
            fig, ax = plt.subplots()
            category_counts.plot(kind='bar', ax=ax)
            ax.set_xlabel("Category")
            ax.set_ylabel("Number of Transactions")
            st.pyplot(fig)

            # Pie chart of spending by category
            st.write("Spending distribution by category:")
            category_sums = df_transactions.groupby("Category")["amount"].sum()
            fig, ax = plt.subplots()
            category_sums.plot(kind='pie', autopct='%1.1f%%', startangle=140, ax=ax)
            ax.set_ylabel("")  # Hide y-label for pie chart
            st.pyplot(fig)

        else:
            st.warning("No transactions were identified.")
    else:
        st.warning("Please upload at least one PDF file.")
    
    end_time = time.time()
    st.write(f"Processing completed in {end_time - start_time:.2f} seconds.")
