import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import json
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Cargar la clave de API
with open("api_key.txt", "r") as file:
    openai_api_key = file.read().strip()

# Configuración del modelo LLM
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    max_tokens=4096,
    openai_api_key=openai_api_key
)

# Definir los schemas de respuesta para el parser
response_schemas = [
    ResponseSchema(name="trans_date", description="Transaction date"),
    ResponseSchema(name="description", description="The transaction description"),
    ResponseSchema(name="amount", description="Amount of the transaction"),
    ResponseSchema(name="currency", description="Currency of the transaction", optional=True),
    ResponseSchema(name="type of transaction", description="Type of the transaction: Debit or Credit", optional=True)
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Definir el prompt template para la extracción de transacciones
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

# Crear la cadena de LLM para la extracción de transacciones
transaction_chain = LLMChain(
    llm=llm,
    prompt=transaction_prompt
)

# Función para extraer texto de un archivo PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Configuración de la interfaz de Streamlit
st.title("PDF Bank Statement Transaction Extractor")
st.write("Upload a PDF bank statement to extract transactions.")

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if st.button("Process PDFs"):
    all_transactions = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Extraer texto del archivo PDF
            extracted_text = extract_text_from_pdf(uploaded_file)
            processed_texts = extracted_text.split("\n")

            # Extracción de transacciones usando LLM
            for idx, text in enumerate(processed_texts):
                transactions_data = transaction_chain.predict(text=text)
                try:
                    parsed_transactions = json.loads(transactions_data)
                    if isinstance(parsed_transactions, list):
                        all_transactions.extend(parsed_transactions)
                except json.JSONDecodeError:
                    st.error(f"Error decoding JSON for part {idx + 1} of {uploaded_file.name}")

        if all_transactions:
            # Convertir los datos de transacciones a un DataFrame de pandas
            df_transactions = pd.DataFrame(all_transactions)

            # Mostrar las transacciones en la app
            st.write("Extracted Transactions")
            st.dataframe(df_transactions)

            # Opción para descargar el archivo Excel
            st.download_button(
                label="Download transactions as Excel",
                data=df_transactions.to_excel(index=False),
                file_name="transactions.xlsx"
            )
        else:
            st.warning("No transactions were identified.")
    else:
        st.warning("Please upload at least one PDF file.")
