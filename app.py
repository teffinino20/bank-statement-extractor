# Import necessary libraries
import pandas as pd
import openai
import matplotlib.pyplot as plt
import seaborn as sns

openai_api_key = st.secrets["openai"]["api_key"]

# Set up OpenAI API key for classification
openai.api_key = openai_api_key

# Load the Excel file with transaction data
df = pd.read_excel('transactions (4).xlsx')
# Define categories
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

# Apply the classification function to the Description column
df['Category'] = df['Description'].apply(classify_transaction_gpt)

# Clean the Amount column by removing special characters
df['Amount'] = df['Amount'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Filter out payments
df = df[~df['Description'].isin(["Payment", "ONLINE PAYMENT - THANK YOU"])]
# Display processed data
print("Processed Data:")
print(df[['Transaction Date', 'Description', 'Amount', 'Category']].head(10))
Processed Data:
  Transaction Date             Description    Amount             Category
0       13/12/2020  ALI*ALIEXPRESS ALIEXPR    -16.46  Shopping & Personal
1       03/12/2020  ALI*ALIEXPRESS ALIEXPR     -1.51  Shopping & Personal
2       03/12/2020  ALI*ALIEXPRESS ALIEXPR     -0.88  Shopping & Personal
3       03/12/2020  ALI*ALIEXPRESS ALIEXPR     -1.27  Shopping & Personal
4       19/11/2020  ABONO SUCURSAL VIRTUAL   -202.00  Miscellaneous/Other
5       15/12/2020    INTERESES CORRIENTES   4250.27  Miscellaneous/Other
6       15/12/2020         CUOTA DE MANEJO  31290.00  Miscellaneous/Other
7       14/12/2020      MERCADOPAGO CABIFY  18975.00       Transportation
8       13/12/2020      MERCADOPAGO CABIFY   9715.00       Transportation
9       13/12/2020      MERCADOPAGO CABIFY  18059.00       Transportation
# Visualization 1: Transaction count by category
plt.figure(figsize=(10, 6))
sns.countplot(x='Category', data=df)
plt.xlabel("Category")
plt.ylabel("Number of Transactions")
plt.title("Transaction Count by Category")
plt.xticks(rotation=45)
plt.show()

# Count the number of transactions for each category
category_counts = df['Category'].value_counts()

# Plotting the pie chart for transaction count distribution by category
plt.figure(figsize=(8, 8))
category_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title("Transaction Count by Category")
plt.ylabel("")  
plt.show()

# Convert 'Transaction Date' to datetime and create columns for Day and Month
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
df['Day'] = df['Transaction Date'].dt.day
df['Month'] = df['Transaction Date'].dt.month

# Create a pivot table for the heatmap
pivot = df.pivot_table(values='Amount', index='Day', columns='Month', aggfunc='sum')
sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".2f")
plt.title("Heatmap of Expenses by Date")
plt.show()
C:\Users\steph\AppData\Local\Temp\ipykernel_6260\294491560.py:2: UserWarning: Parsing dates in DD/MM/YYYY format when dayfirst=False (the default) was specified. This may lead to inconsistently parsed dates! Specify a format to ensure consistent parsing.
  df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])

# Calculate running balance and plot
df['Running Balance'] = df['Amount'].cumsum()
plt.figure(figsize=(12, 6))
plt.plot(df['Transaction Date'], df['Running Balance'], marker='o')
plt.title("Running Balance Over Time")
plt.xlabel("Date")
plt.ylabel("Running Balance")
plt.xticks(rotation=45)
plt.show()

import matplotlib.dates as mdates

# Convert 'Transaction Date' to datetime if it's not already
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])

# Plot Date vs. Amount with color for Category
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Transaction Date', y='Amount', hue='Category', palette="Set2", s=100, alpha=0.7)
plt.title("Scatter Plot of Transactions by Date and Amount")
plt.xlabel("Transaction Date")
plt.ylabel("Amount")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(title="Category")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))  # Format date
plt.show()

# Calculate monthly total and plot for forecasting
df['Month'] = df['Transaction Date'].dt.to_period("M")
monthly_spending = df.groupby('Month')['Amount'].sum()

# Plot monthly spending with a rolling average
plt.figure(figsize=(10, 5))
monthly_spending.plot(kind='line', label="Actual Spending")
monthly_spending.rolling(window=3).mean().plot(label="3-Month Rolling Avg", linestyle='--')
plt.title("Monthly Spending Forecast")
plt.xlabel("Month")
plt.ylabel("Spending")
plt.legend()
plt.show()

# Group by Category and Type of Transaction to get total credits and debits per category
transaction_sums = df.groupby(["Category", "Type of Transaction"])["Amount"].sum().unstack(fill_value=0)

# Plotting the bar chart to show credits and debits per category
plt.figure(figsize=(12, 8))
transaction_sums.plot(kind='bar', stacked=True)
plt.xlabel("Category")
plt.ylabel("Total Amount")
plt.title("Total Credits and Debits per Category")
plt.xticks(rotation=45)
plt.legend(title="Transaction Type")
plt.tight_layout()
plt.show()
<Figure size 1200x800 with 0 Axes>

# Count the number of transactions for each type (Credit and Debit)
transaction_type_counts = df['Type of Transaction'].value_counts()

# Plot the pie chart showing the percentage of credit vs debit transactions
plt.figure(figsize=(8, 8))
transaction_type_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, labels=["Debit", "Credit"], colors=["salmon", "skyblue"])
plt.title("Percentage of Debit vs Credit Transactions")
plt.ylabel("")  # Hide y-axis label for a cleaner look
plt.show()

# Create a DataFrame with relevant columns for the transaction summary
transaction_summary = df[['Transaction Date', 'Description', 'Amount', 'Category']]

# Display the table to review transactions, values, and categories
print("Transaction Summary Table:")
transaction_summary.head(30)  # Display the first 10 rows as a preview
