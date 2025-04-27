import os
import io
import contextlib
import re
from dotenv import find_dotenv, load_dotenv

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import sklearn as sk
import statsmodels.api as sm
import statsmodels.formula.api as smf

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load RAG vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    "vectorstore/faiss_index", embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_type="similarity", k=4)

# Title
st.title("AI-Powered Data Analysis Assistant")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")
question = st.text_input("What would you like to see using this data?")

#Checkboxes
show_rag_context = st.checkbox("Show RAG context")
require_plot = st.checkbox("Always include a plot in the analysis")


# Initialize OpenAI LLM
template = """
You are a Python data analysis assistant with a specialization in stock market analysis. 
Here is expert context from epidemiological data analysis resources:

{context_text}

A user has uploaded the following dataset sample:

{data_sample}

User input: "{question}" 

Please determine the most visual appropriate representation, explain the method, and generate Python code using the uploaded dataset, which is already loaded in a variable called 'df'.

Please generate graphs. Include both univariate and multivariate graphs. After the code for these graphs, please also include a statistical test or regression analysis to determine how significant the relationship you are visualizing is.
Use print() to output a short description for the user to interpret the results of the tests as well.

Ensure that any necessary data preprocessing is completed before any subsequent code. You may have preprocessed data or you may not.
#


"""
prompt = PromptTemplate(
    input_variables=["data_sample", "question", "context_text", "var_info"],
    template=template,
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, api_key=openai_api_key)
llm_chain = prompt | llm

# Main logic
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)

    st.write("Preview of your data:")
    st.dataframe(df.head())
    st.write(f"Your dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

    if st.checkbox("Show all columns"):
        st.write("Columns:", list(df.columns))

    if st.checkbox("Show summary statistics"):
        st.write(df.describe(include="all"))

    # Save to session
    st.session_state.df = df

if question and st.button("Generate Analysis"):
    df = st.session_state.get("df")
    data_sample = df.head(10).to_csv(index=False)

    docs = retriever.invoke(question)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    col_types = df.dtypes.apply(str).to_dict()
    categorical_vars = [col for col, dtype in col_types.items() if dtype == "object"]
    numeric_vars = [
        col for col, dtype in col_types.items() if dtype in ["int64", "float64"]
    ]
    var_info = f"""
Your dataset contains:
- Categorical: {", ".join(categorical_vars) or "None"}
- Numeric: {", ".join(numeric_vars) or "None"}
"""

    modified_question = question
    if require_plot:
        modified_question += " Please include at least one appropriate plot."

    result = llm_chain.invoke(
        {
            "data_sample": data_sample,
            "question": modified_question,
            "context_text": context_text,
            "var_info": var_info,
        }
    )

    output_text = result.content if hasattr(result, "content") else result
    st.session_state.generated_code = output_text
    st.session_state.generated_response = output_text
    st.session_state.context_text = context_text
    st.session_state.question = question

# Show result and RAG context
if "generated_response" in st.session_state:
    st.markdown("### Suggested Analysis & Code")
    st.markdown(st.session_state.generated_response)

if show_rag_context and "context_text" in st.session_state:
    st.markdown("### Retrieved RAG Context")
    st.markdown(st.session_state.context_text)

# Code Execution
if "generated_code" in st.session_state:
    if st.button("Run the code"):
        result = st.session_state.generated_code

        # Extract Python code block from the result
        code_match = re.search(r"```python(.*?)```", result, re.DOTALL)
        code_to_run = code_match.group(1).strip() if code_match else result

        st.markdown("### Code Output")

        # Capture stdout
        f = io.StringIO()

        # Reset any existing figures
        plt.close("all")

        # Namespace for execution
        local_namespace = {"df": df, "pd": pd, "np": np, "plt": plt, "sns": sns}


        with contextlib.redirect_stdout(f):
            try:
                # Run the generated code
                exec(code_to_run, local_namespace)

                # Capture all output text
                output_text = f.getvalue()

                # Capture all generated figures
                figures = [plt.figure(i) for i in plt.get_fignums()]

                # Create tabs dynamically based on what's generated
                tabs = []
                if output_text.strip():
                    tabs.append("Text Output")
                for i, _ in enumerate(figures):
                    tabs.append(f"Figure {i + 1}")

                # Display results in tabs
                if tabs:
                    tab_objs = st.tabs(tabs)
                    tab_index = 0

                    if output_text.strip():
                        with tab_objs[tab_index]:
                            st.text(output_text)
                        tab_index += 1

                    for fig in figures:
                        with tab_objs[tab_index]:
                            st.pyplot(fig)
                        tab_index += 1
                else:
                    st.info("No output generated.")

            except Exception as e:
                st.error(f"Error running code: {e}")