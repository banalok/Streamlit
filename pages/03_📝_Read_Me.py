import streamlit as st

def load_readme():
    with open("READ_TEXT.md", "r", encoding="utf-8") as file:
        readme_text = file.read()
    return readme_text

readme_text = load_readme()
st.markdown(readme_text)