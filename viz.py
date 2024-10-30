import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import json
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import matplotlib.pyplot as plt
import seaborn as sns
import io
import time

# Access OpenAI API key from Streamlit's secrets
openai_api_key = st.secrets["openai"]["api_key"]

# Configure LLM model
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    max_tokens=4096,
    openai_api_key=openai_api_key
)

# Define response schemas for transaction extraction
response_schemas = [
    ResponseSchema(name="trans_date", description="Transaction date"),
    ResponseSchema(name="description", description="The transaction description"),
    ResponseSchema(name="amount", description="Amount of the transaction"),
    ResponseSchema(name="currency", description="Currency of the transaction", optional=True),
    ResponseSchema(name="type of transaction", description="Type of the transaction: Debit or Credit", optional=True)
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Define prompt template for transaction extraction
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

# Create LLM chain for transaction extraction
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
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# Prompt-engineered GPT classification function
categories = [
    "Food & Dining", "Utilities", "Transportation", "Entertainment & Recreation",
    "Shopping & Personal", "Healthcare", "Business Expenses", "Home & Rent",
    "Education", "Insurance", "Loan Payments", "Gifts & Donations",
    "Professional Services", "Taxes", "Miscellaneous/Other"
]

def classify_transaction_gpt(description):
    prompt = (
        f"Classify the following bank transaction description into one of these categories: {', '.join(categories)}.\n"
        "Here are some examples to guide the classification:\n"
        "- Transactions mentioning 'Chevron', 'Exxon', or 'Shell' are typically categorized as 'Transportation'.\n"
        "- Transactions with terms like 'Grocery', 'Food', or 'Restaurant' go under 'Food & Dining'.\n"
        "- Descriptions mentioning 'Utilities', 'Electric', or 'Gas Bill' are categorized as 'Utilities'.\n"
        "- Purchases from places like 'Amazon' or 'Walmart' are usually 'Shopping & Personal'.\n"
        "- Hotel or travel-related descriptions often fall under 'Entertainment & Recreation'.\n\n"
        f"Now, classify this transaction:\n\nDescription: '{description}'"
    )
    
    try:
        response = llm.ChatCompletion.create(
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

# Streamlit interface
st.title("Bank Statement Extractor")
st.write("Upload a PDF bank statement to extract transactions and generate visualizations.")

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
                    continue

        if all_transactions:
            # Convert transaction data to a DataFrame and standardize column names
            df = pd.DataFrame(all_transactions)
            df.columns = df.columns.str.lower()

            # Verify necessary columns
            if 'description' not in df.columns:
                st.warning("The 'description' column is missing in the data.")
            else:
                # Clean and classify transactions
                df = df[~df['description'].isin(["Payment", "ONLINE PAYMENT - THANK YOU"])]
                df['category'] = df['description'].apply(classify_transaction_gpt)
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

                # Display extracted and classified transactions
                st.write("Extracted and Classified Transactions")
                st.dataframe(df)

                # Save transactions to Excel
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False)
                    buffer.seek(0)

                st.download_button(
                    label="Download transactions as Excel",
                    data=buffer,
                    file_name="transactions.xlsx"
                )

                # Visualization
                st.subheader("Visualizations")

                # Pie chart of transaction count by category
                category_counts = df['category'].value_counts()
                plt.figure(figsize=(8, 8))
                category_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
                plt.title("Transaction Count by Category")
                st.pyplot(plt)

                # Debit vs. Credit count
                transaction_type_counts = df['type of transaction'].value_counts()
                plt.figure(figsize=(8, 6))
                sns.barplot(x=transaction_type_counts.index, y=transaction_type_counts.values)
                plt.title("Count of Debit vs Credit Transactions")
                plt.xlabel("Type of Transaction")
                plt.ylabel("Count")
                st.pyplot(plt)

                # Scatter plot of transactions by category and amount
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=df, x="category", y="amount", hue="category", palette="tab10", s=100)
                plt.title("Scatter Plot of Transactions by Category and Amount")
                plt.xticks(rotation=45)
                st.pyplot(plt)

                # Running balance over time
                df['transaction date'] = pd.to_datetime(df['transaction date'], errors='coerce')
                df = df.sort_values('transaction date')
                df['running_balance'] = df['amount'].cumsum()
                plt.figure(figsize=(10, 6))
                sns.lineplot(data=df, x="transaction date", y="running_balance")
                plt.title("Running Balance Over Time")
                plt.xlabel("Date")
                plt.ylabel("Balance")
                st.pyplot(plt)

        else:
            st.warning("No transactions were identified.")
    else:
        st.warning("Please upload at least one PDF file.")
    
    end_time = time.time()
    st.write(f"Processing completed in {end_time - start_time:.2f} seconds.")


    
