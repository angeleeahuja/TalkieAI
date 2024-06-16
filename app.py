import os
import fitz
import mimetypes
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

load_dotenv()
os.getenv('GOOGLE_API_KEY')
genai.configure(api_key= os.getenv('GOOGLE_API_KEY'))

st.set_page_config(page_title= 'TalkieAI', 
                   page_icon= '🤖',
                   layout= 'wide')
st.sidebar.title('TalkieAI')

model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
chat = model.start_chat(history=[])

prompt = '''
    You are a friendly assisstant. Your job is to respond to user 
    queries and provide helpful information. You can hypothetically 
    answer question to the best of your knowledge. Don't leave any 
    answer empty.
'''

def get_chat_response(question):
    response = chat.send_message(question, stream=True)
    response.resolve()
    return response

def get_image_chat_response(input, image, prompt):
    response = model.generate_content([input, image[0], prompt])
    return response.text

def image_details(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [{
            'mime_type' : uploaded_file.type,
            'data' : bytes_data
        }]
        return image_parts
    else:
        raise FileNotFoundError('No file uploaded')
    
def get_text_from_pdf(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def get_text_chunk(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 10000, chunk_overlap= 1000)
    chunk = text_splitter.split_text(text)
    return chunk
    
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model= 'models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks, embedding= embeddings)
    vector_store.save_local('faiss_index')

def get_conversational_chain():
    prompt_template = """
        Answer the question as detailed as possible from the provided context, 
        make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", 
        don't provide the wrong answer
        
        Context: {context}
        Question: {question}

        Answer:
    """
    model = ChatGoogleGenerativeAI(model='gemini-pro')
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local('faiss_index', embeddings=embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {'input_documents': docs, 'question': user_question},
        return_only_outputs= True
    )
    return response


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there, How can I help you today?"}
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_prompt = st.chat_input("Type your message here...")

uploaded_file = st.sidebar.file_uploader('Upload Files', type= ['jpg', 'jpeg', 'png', 'pdf'])
uploaded_image = None 
uploaded_pdf = None
if uploaded_file is not None:
    mime_type = mimetypes.guess_type(uploaded_file.name)[0]
    if mime_type and mime_type.startswith('image'):
        uploaded_image = uploaded_file
        image = Image.open(uploaded_image)
        st.sidebar.image(image, use_column_width=True)
    elif mime_type == 'application/pdf':
        uploaded_pdf = uploaded_file
        with st.sidebar:
            raw_text = get_text_from_pdf(uploaded_pdf)
            text_chunks = get_text_chunk(raw_text)
            get_vector_store(text_chunks)
            st.success("Done")
    else:
        st.sidebar.write("Unsupported file type.")

if user_prompt:
    user_message = f"{user_prompt}"
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.write(user_message)
    if uploaded_image is not None:
        image_data = image_details(uploaded_image)
        answer = get_image_chat_response(user_prompt, image_data, prompt)
        assistant_message = f"{answer}"
    elif uploaded_pdf is not None:
        answer = user_input(user_prompt)
        assistant_message = f"{answer['output_text']}"
    else:
        answer = get_chat_response(user_prompt)
        assistant_message = f"{answer.text}"

    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
    with st.chat_message("assistant"):
        st.write(assistant_message)

for query, response in st.session_state.chat_history:
    st.write(f"{query}: {response}")