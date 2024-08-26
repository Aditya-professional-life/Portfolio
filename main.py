import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about the resume."]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about the resume", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def create_conversational_chain(vector_store):
    # Create LLM with Replicate
    llm = Replicate(
        streaming=True,
        model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
        callbacks=[StreamingStdOutCallbackHandler()],
        input={"temperature": 0.01, "max_length": 500, "top_p": 1}
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )
    return chain

def main():
    load_dotenv()
    initialize_session_state()
    
    # Sidebar navigation
    st.sidebar.title("Portfolio Navigator")
    section_choice = st.sidebar.radio(
        "Select a section",
        ["Introduction", "Skills", "Projects", "Work Experience", "Get in Touch", "Resume ChatBot"]
    )

    # Introduction section
    if section_choice == "Introduction":
        st.title("Introduction")
        st.write("""
        Hello,
        I'm Aditya Kumar!
        NMIT BANGALORE '25, B.E Student.

        I am a motivated and dedicated college student with a strong passion for artificial intelligence (AI) and Data Science (DS). My interests include Artificial Intelligence, Machine Learning, Computer Vision, and Natural Language Processing.
        """)
        st.download_button(
            label="Download Resume",
            data=open("Resume.pdf", "rb").read(),
            file_name="Aditya_Kumar_Resume.pdf",
            mime="application/pdf"
        )

    # Skills section
    elif section_choice == "Skills":
        st.title("Skills")
        st.write("""
        **Core Subjects:** Operating Systems (OS), Database Management Systems (DBMS), Computer Networks (CN), Cloud Computing

        **Programming:** Python, SQL, Numpy, Matplotlib, Seaborn, Streamlit, Selenium, Statistics, Plotly

        **Machine Learning:** Data Preprocessing, Feature Engineering, Model Selection, Model Building, Model Training, Hyperparameter Tuning, Supervised Learning, Unsupervised Learning

        **Deep Learning:** ANN, CNNs, RNNs, PyTorch, Transfer Learning, Model Optimization

        **Computer Vision:** OpenCV, YOLO, Object Detection, Real-time Processing

        **MLOps:** Model Deployment, Continuous Integration/Continuous Deployment (CI/CD), Model Versioning, Automation, Containerization (Docker)
        """)

    # Projects section
    elif section_choice == "Projects":
        st.title("Projects")
        st.subheader("LangChain PDF Interact")
        st.write("""
        **Technologies:** Streamlit, LangChain, PyPDF2, Hugging Face Transformers, FAISS, Replicate
        - Built an advanced interactive chatbot application using Streamlit, LangChain, and Hugging Face Transformers for sophisticated conversation and in-depth document analysis.
        - Enabled seamless multi-format document processing with PyPDF2, Docx2txt, and TextLoader, handling PDFs, DOCX, and TXT files efficiently.
        - Utilized FAISS and HuggingFaceEmbeddings to enable high-speed similarity search and create detailed document embeddings across large text datasets.
        - Designed a highly intuitive user interface for natural language queries, delivering precise and relevant responses based on comprehensive document content.
        [GitHub Repository](https://github.com/Aditya-professional-life/llama-2-chatbot)
        """)

        st.subheader("Multi-Modal AI Voice Assistant")
        st.write("""
        **Technologies:** Groq, Google Generative AI, Whisper, OpenCV, pyttsx3, pyperclip, PIL
        - Developed a sophisticated multi-modal AI assistant that integrates Groq and Google Generative AI, delivering advanced text and image-based responses to user inputs.
        - Implemented comprehensive image processing functionalities, including screenshot capture and webcam image analysis, to provide detailed context and enhance the assistant’s response accuracy.
        - Facilitated clipboard text extraction and seamlessly integrated it into the assistant’s responses, enriching user interactions with additional relevant information.
        - Designed a versatile user interface that supports both text input and manual command selection, allowing for a flexible and interactive experience with the AI assistant.
        [GitHub Repository](https://github.com/Aditya-professional-life/AI-Multi-Modal-Assistant)
        """)

        st.subheader("Real-time Object Detection & Tracking")
        st.write("""
        **Technologies:** OpenCV, Tracker
        - Engineered real-time object tracking with Python, OpenCV, and Euclidean distance tracking.
        - Applied advanced computer vision techniques like background subtraction and contour detection.
        - Demonstrated Python proficiency and algorithmic optimization to enhance object tracking performance.
        [GitHub Repository](https://github.com/Aditya-professional-life/Real-Time-Object-Tracking-System)
        """)

    # Work Experience section
    elif section_choice == "Work Experience":
        st.title("Work Experience")

        st.subheader("BCG GenAI Intern (July 2024 – Aug 2024)")
        st.write("""
        - Completed a job simulation involving AI-powered financial chatbot development for BCG's GenAI Consulting team.
        - Gained experience in Python programming, including the use of libraries such as pandas for data manipulation.
        - Integrated and interpreted complex financial data from 10-K and 10-Q reports, employing rule-based logic to create a chatbot that provides user-friendly financial insights and analysis.
        """)

        st.subheader("Cognizant AI Intern (June 2024 – July 2024)")
        st.write("""
        - Completed a job simulation focused on AI for Cognizant's Data Science team.
        - Conducted exploratory data analysis using Python and Google Colab for one of Cognizant's technology-led clients, Gala Groceries.
        - Prepared a Python module that contains code to train a model and output the performance metrics for the Machine Learning engineering team.
        - Communicated findings and analysis in the form of a PowerPoint slide to present the results back to the business.
        """)

        st.subheader("Ineuron.ai Machine Learning Intern (March 2024 – June 2024)")
        st.write("""
        - Conducted comprehensive Exploratory Data Analysis (EDA) on the Zomato Dataset.
        - Developed and implemented machine learning models for accurate rating prediction.
        - Utilized diverse algorithms to tailor a robust predictive model specific to the Zomato Dataset.
        - Enhanced accuracy in restaurant rating forecasts, contributing to strategic business initiatives.
        """)
        # Get in Touch section
    elif section_choice == "Get in Touch":
        st.title("Get in Touch with Me")
        with st.form(key='contact_form'):
            st.write("Feel free to reach out to me through the form below.")
            name = st.text_input("Name")
            email = st.text_input("Email")
            message = st.text_area("Message")
            submit_button = st.form_submit_button("Send Message")

            if submit_button:
                if name and email and message:
                    st.success("Thank you for your message! I will get back to you soon.")
                else:
                    st.error("Please fill out all fields before submitting.")

    # Resume ChatBot section
    elif section_choice == "Resume ChatBot":
        st.title("Resume ChatBot using LLaMA2 :books:")
        
        st.write("Please enter your query about the resume below:")
        user_query = st.text_input("Your Question", placeholder="Ask about the resume")

        if user_query:
            st.spinner('Processing your query...')
            
            # Process and respond to user query
            file_path = "Resume.pdf"  # Update with the path to your resume file
            
            file_extension = os.path.splitext(file_path)[1]
            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension == ".docx" or file_extension == ".doc":
                loader = Docx2txtLoader(file_path)
            elif file_extension == ".txt":
                loader = TextLoader(file_path)

            if loader:
                text = loader.load()
                
                text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
                text_chunks = text_splitter.split_documents(text)

                # Create embeddings
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

                # Create vector store
                vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

                # Create the chain object
                chain = create_conversational_chain(vector_store)

                response = conversation_chat(user_query, chain, st.session_state['history'])
                st.write(response)

if __name__ == "__main__":
    main()
