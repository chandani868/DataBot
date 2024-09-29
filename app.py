import os
import pandas as pd
import streamlit as st
from pandasai import SmartDataframe, SmartDatalake
from pandasai.llm import GooglePalm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
# from pandasai.llm.openai import OpenAI
from langchain_groq import ChatGroq


def read_csv_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        st.warning("UTF-8 encoding failed. Trying latin1 encoding.")
        try:
            df = pd.read_csv(uploaded_file, encoding='latin1')
            print(uploaded_file)
        except UnicodeDecodeError:
            st.error("Unable to read the file with both UTF-8 and ISO-8859-1 encodings.")
            return None
    return df


def main():
    st.set_page_config(page_title="Smart Data Query App", layout="wide")

    st.title("Smart Data Query App")

    # Choice to upload one or two CSV files
    file_count = st.radio("How many CSV files would you like to upload?", (1, 2))

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file1 = st.file_uploader("Choose the first CSV file", type="csv", key="file1")
    if file_count == 2:
        with col2:
            uploaded_file2 = st.file_uploader("Choose the second CSV file", type="csv", key="file2")
    else:
        uploaded_file2 = None

    if uploaded_file1 is not None or uploaded_file2 is not None:
        df1 = read_csv_file(uploaded_file1) if uploaded_file1 is not None else None
        df2 = read_csv_file(uploaded_file2) if uploaded_file2 is not None else None

        if df1 is not None or df2 is not None:
            if df1 is not None:
                with st.expander("Preview First CSV File"):
                    st.dataframe(df1.head())
            if df2 is not None:
                with st.expander("Preview Second CSV File"):
                    st.dataframe(df2.head())

            pandas_api = os.environ['PANDASAI_API_KEY']
            google_api = os.environ['GOOGLE_API_KEY']
            groq_api = os.environ['GROQ_API_KEY']

            # Set up the ChatGroq model
            llm = ChatGroq(
                groq_api_key=groq_api,
                model_name='mixtral-8x7b-32768'
            )
            # llm = GooglePalm(api_key=google_api)

            if df1 is not None and df2 is not None:
                lake = SmartDatalake([df1, df2])
            else:
                lake = SmartDataframe(df1, config = {"LLM": llm, "conversational": True, "verbose": True}) if df1 is not None else None

            datalake_1 = lake
            query = st.text_input("Enter your query:")

            submitted = st.button("Submit")

            if submitted:
                if query:
                    response = datalake_1.chat(query)

                    if "Unfortunately, I was not able to answer your question, because of the following error:" in response:
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-pro", verbose=True, google_api_key=google_api
                        )

                        agent = create_pandas_dataframe_agent(
                            llm, 
                            df1,
                            verbose=True,
                        )

                        response2 = agent.invoke(query)
                        st.write("Response:")
                        st.write(response2["output"])

                    else:
                        st.write("Response:")
                        st.write(response)

                    # Append the query and response to the session state for query history
                    if 'query_history' not in st.session_state:
                        st.session_state.query_history = []
                    st.session_state.query_history.append((query, response))
                else:
                    st.write("Please enter a query.")

            # Display query history
            if 'query_history' in st.session_state and st.session_state.query_history:
                st.subheader("Query History")
                for q, r in st.session_state.query_history:
                    st.write(f"**Query:** {q}")
                    st.write(f"**Response:** {r}")
                    st.write("---")

        else:
            st.error("Failed to read one or both CSV files. Please check the files and try again.")
    else:
        st.write("Please upload at least one CSV file.")


if __name__ == "__main__":
    main()
