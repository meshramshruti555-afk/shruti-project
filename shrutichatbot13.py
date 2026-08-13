import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked."),
        ("user", "question: {question}")
    ]
)

# Streamlit UI
st.title("My GPT")
input_text = st.text_input("Ask your question:")

# Ollama model
llm = OllamaLLM(model="gemma2:2b")

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Only run chain if user provides input
if input_text:
    try:
        with st.spinner("Thinking..."):
            response = chain.invoke({"question": input_text})
        st.markdown(response)
    except Exception as e:
        st.error(f"Error: {e}")
