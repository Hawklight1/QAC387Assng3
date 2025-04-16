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

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Title
st.title("AI-Powered Data Analysis Assistant")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")
question = st.text_input("What do you hypothesize about this data?")


# Initialize OpenAI LLM
template = """
You are a Python data analysis assistant. A user has uploaded the following dataset sample:

{data_sample}

The user hypothesizes: "{question}" 

Please determine the most visual representation, explain the method, and generate Python code using the uploaded dataset, which is already loaded in a variable called 'df'.

Please generate graphs only using matplotlib. Include both univariate and multivariate graphs. After the code for these graphs, please also include a statistical test or regression analysis to determine how significant the relationship you are visualizing is.
Use print() to output a short description for the user to interpret the results of the tests as well.
#


"""
prompt = PromptTemplate(input_variables=["data_sample", "question"], template=template)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_chain = prompt | llm

# Main logic
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Fix Arrow serialization issues
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)

    st.write("Preview of your data:")
    st.dataframe(df.head())
    st.write(f"Your dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

    if st.checkbox("Show all columns"):
        st.write("Columns:", list(df.columns))

    if st.checkbox("Show full dataset"):
        st.dataframe(df)

    if st.checkbox("Show summary statistics"):
        st.write(df.describe(include="all"))
    

    if question and st.button("Generate Analysis"):
        with st.spinner("Working on it..."):
            data_sample = df.head(10).to_csv(index=False)
            result = llm_chain.invoke(
                {"data_sample": data_sample, "question": question}
            )

        st.markdown("### Suggested Analysis & Code")
        output_text = result.content if hasattr(result, "content") else result
        st.markdown(output_text if output_text else "*No response received from LLM*")
        st.session_state.generated_code = output_text

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