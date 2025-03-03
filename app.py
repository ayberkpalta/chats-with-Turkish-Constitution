import os
import warnings
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Uyarıları kapat
warnings.filterwarnings("ignore", category=DeprecationWarning)

# OpenAI API anahtarını ayarla (Kendi anahtarını eklemelisin)
os.environ["OPENAI_API_KEY"] = "enter_api_key"

# PDF'den Anayasa metnini yükle
loader = PyPDFLoader("anayasa_eng.pdf")
documents = loader.load()

# Metni küçük parçalara böl
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# Chroma vektör veritabanını oluştur
vectorstore = Chroma.from_documents(splits, OpenAIEmbeddings())

# Retriever oluştur
retriever = vectorstore.as_retriever()

# Açık kaynaklı LLM (Mistral) kullanımı
llm = Ollama(model="mistral")

### --- Zero-Shot Prompting ---
zero_shot_prompt = PromptTemplate(
    input_variables=["question"],
    template="Soru: {question}\nCevap:"
)

### --- Few-Shot Prompting ---
# Örnek soru-cevaplar oluştur
examples = [
    {"question": "Türkiye Cumhuriyeti'nin resmi dili nedir?", "answer": "Türkiye Cumhuriyeti'nin resmi dili Türkçedir."},
    {"question": "Cumhurbaşkanı kaç yıl görev yapar?", "answer": "Cumhurbaşkanı 7 yıl süreyle görev yapar."},
    {"question": "Türkiye'nin başkenti neresidir?", "answer": "Türkiye'nin başkenti Ankara'dır."}
]

# Few-Shot Prompt Template
example_template = """
Soru: {question}
Cevap: {answer}
"""
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=PromptTemplate(input_variables=["question", "answer"], template=example_template),
    suffix="Soru: {question}\nCevap:",
    input_variables=["question"]
)

# Kullanıcıdan aldığı inputa göre uygun prompting tekniğini belirleyen fonksiyon
def get_prompting_type(question, few_shot=True):
    if few_shot:
        return few_shot_prompt.format(question=question)
    else:
        return zero_shot_prompt.format(question=question)

# RetrievalQA zinciri oluştur
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Sohbet geçmişi için Streamlit session state
if "history" not in st.session_state:
    st.session_state.history = []

# Kullanıcı ve model mesajlarını saklama fonksiyonu
def add_to_history(user_message, bot_response):
    st.session_state.history.append({"role": "user", "content": user_message})
    st.session_state.history.append({"role": "assistant", "content": bot_response})

# Geçmiş sohbeti almak için fonksiyon
def get_conversation_context():
    return [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.history]

# LLM ile sohbet fonksiyonu
def chat_with_llm(user_input, few_shot=True):
    # Zero-shot veya Few-shot prompting'i seç
    prompt = get_prompting_type(user_input, few_shot=few_shot)

    # LLM'den yanıt al
    response = qa_chain.invoke(prompt)

    # Geçmişi güncelle
    add_to_history(user_input, response["result"])

    return response["result"]

# ---- Streamlit UI ----
st.title("Turkish Constitution Chatbot 🇹🇷📜")

# Kullanıcıdan Few-Shot mı, Zero-Shot mı kullanacağını seçmesini iste
few_shot_toggle = st.checkbox("Few-Shot Prompting Kullan", value=True)

# Sohbet geçmişini göster
for message in st.session_state.history:
    role = "You" if message["role"] == "user" else "Bot"
    st.text_area(role, value=message["content"], height=75, disabled=True)

# Kullanıcı girişi
user_input = st.text_input("You: ", key="input")

# Gönder butonu
if st.button("Send"):
    if user_input.lower() == "exit":
        st.stop()
    else:
        response = chat_with_llm(user_input, few_shot=few_shot_toggle)
        st.rerun()  
