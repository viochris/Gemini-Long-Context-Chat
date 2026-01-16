import time
import streamlit as st
from google import genai
from google.genai import types

# --- 1. SESSION STATE MANAGEMENT ---
def reset_state():
    """Clear chat history to start a fresh conversation session."""
    st.session_state["message"] = []

# Initialize Session State variables if they don't exist
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""
if "message" not in st.session_state:
    st.session_state["message"] = []
if "document" not in st.session_state:
    st.session_state["document"] = None

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="DocuChat AI", page_icon="📚", layout="wide")
st.title("🤖 DocuChat: Chat with your Documents")
st.markdown(
    """
    **Unlock knowledge from your files.** Upload PDFs, Text files, or Markdown notes sidebar to start analyzing.
    """
)

# --- 3. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ System Configuration")

    # Securely capture API Key
    google_api_key = st.text_input("🔑 Enter Google API Key", type="password", key="input_widget")
    st.session_state["api_key"] = google_api_key

    # Language Selection Widget
    # This variable stores the user's choice to be injected into the system prompt later
    target_language = st.selectbox(
        "🌐 Response Language",          # Label with an icon for better visibility
        options=["English", "Indonesian"], # List of available languages for the AI's output
        index=0,                         # Default to "English" (index 0) to ensure the prompt is never empty/None
        help="The AI will answer your questions in this selected language, regardless of the document's language." # Tooltip to explain functionality
    )

    # Button to clear history
    st.button("🔄 Clear Conversation", on_click=reset_state)

    st.divider()

    # File Uploader Widget
    uploaded_files = st.file_uploader(
        label="📂 Upload Knowledge Base",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, MD. The AI will analyze text content from these files."
    )

    # Feedback on file upload status
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} File(s) Ready for Analysis")
    else:
        st.info("ℹ️ Please upload files to begin.")

    st.divider()

    # --- 1. HOW TO USE  ---
    with st.expander("📖 How to Use"):
        st.markdown(
            """
            **Step 1: System Setup**
            Get your API Key from Google AI Studio and enter it securely above.

            **Step 2: Upload Documents**
            Upload PDF, TXT, or MD files.   
            *Tip: You can upload multiple files to create a comprehensive knowledge base.*  
            *⚠️ Recommendation: Upload 1-5 files at a time to ensure optimal speed and avoid timeouts. Exceeding this limit may cause the application to freeze or fail to process the request, rapidly drain your token quota, and confuse the LLM (especially if the topics are different). Stick to focused batches of documents.*

            **Step 3: Select Language**
            Choose "Indonesian" or "English" for the AI's response language.
            
            **Step 4: Start Chatting**
            Ask questions about specific details, summaries, or comparisons between files.
            
            **Step 5: Switching Topics (Crucial!)**
            When moving to a new subject (e.g., from Law to Biology):
            1. Remove old files (click 'x' on file).
            2. **Click 'Reset Conversation'**.
            3. Upload new files.
              
            *Why Reset?* Deleting files is not enough. The AI still remembers the **Chat History**. If you don't reset, it will hallucinate information from the deleted files.  
            *Why Start Fresh?* Mixing unrelated files (e.g., Law + Math) confuses the model, wastes tokens/energy, and slows down the app.

            **💡 Pro Tips:**
            * *Specific:* "Summarize the methodology in the PDF."
            * *Comparative:* "What is the difference between file A and file B?"
            * *Study Aid:* "Generate 10 practice questions with answers to test my understanding."
            * *Clean Slate:* Always click **Reset Conversation** when switching topics!
            """
        )

    # --- 2. FAQ (FORMATTED MARKDOWN) ---
    with st.expander("❓ FAQ"):
        st.markdown(
            """
            **Q: What is an LLM?**  
            A: LLM (Large Language Model) is an AI trained on massive amounts of text to understand and generate human-like language. This app uses Google's **Gemini 2.5 Flash** model to analyze your documents.

            **Q: Why does it discuss old/deleted files?**  
            A: The AI remembers the *entire conversation history*. If you upload new files but don't clear the history, the AI might still refer to the old context.
            **Solution:** Always click the **"🔄 Clear Conversation"** button when starting a new topic or switching files.

            **Q: Is my API Key safe?**  
            A: Yes. It is processed in temporary session memory and is never stored on any server/database.

            **Q: Why is it sometimes slow?**  
            A: This app uses a "Full Context" method. It re-reads every page of your documents for *every* question to ensure maximum accuracy (no hallucination).

            **Q: Can I upload scanned PDFs?**  
            A: Yes, Gemini 2.5 Flash has Vision capabilities and can read text from images/scans.

            **Q: What is the file size limit?**  
            A: Streamlit defaults to 200MB per file, which is sufficient for most large e-books or reports.
            """
        )

# Warning if API Key is missing to prevent errors
if not st.session_state["api_key"]:
    st.warning("⚠️ Access Denied: Please provide a valid Google API Key in the sidebar.")

# --- 4. DISPLAY CHAT HISTORY ---
# Iterate through the session state and render previous messages
for msg in st.session_state["message"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg['content'])

# --- 5. MAIN CHAT INPUT & LOGIC ---
# Capture user input from the chat bar
prompt_text = st.chat_input(placeholder="Ask specific questions about your documents...")

if prompt_text:
    # A. Display User Message immediately in the UI
    st.session_state["message"].append({"role": "human", "content": prompt_text})
    with st.chat_message("human"):
        st.markdown(prompt_text)

    # B. AI Processing Logic
    try:
        # Status container to show the backend process steps (UX improvement)
        with st.status("🚀 Orchestrating AI Workflow...", expanded=True) as status:
            
            # Initialize Gemini Client
            client = genai.Client(api_key=st.session_state["api_key"])
            content_parts = []

            # --- STEP 1: DOCUMENT INGESTION ---
            # Converting uploaded files into a format Gemini can understand
            st.write("📂 Parsing and digitizing document content...")
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    # Normalize filename to handle case sensitivity
                    filename = uploaded_file.name.lower()
                    
                    if filename.endswith('.pdf'):
                        # Handle PDF: Send as Binary Data (Blob)
                        content_parts.append(
                            types.Part.from_bytes(
                                data=uploaded_file.getvalue(),
                                mime_type='application/pdf',
                            )
                        )
                    elif filename.endswith(".txt") or filename.endswith('.md'):
                        # Handle Text/MD: Decode bytes to String and wrap in tags
                        text_data = uploaded_file.getvalue().decode("utf-8")
                        
                        # Formatting for clearer AI understanding (XML-like structure)
                        text_part = f"""
                        === START OF FILE: {uploaded_file.name} ===
                        {text_data}
                        === END OF FILE ===
                        """
                        content_parts.append(text_part)
            
            time.sleep(0.5) # Small delay for visual effect

            # --- STEP 2: CONTEXT RETRIEVAL ---
            # Injecting past conversation history so the AI remembers context
            st.write("🧠 Synchronizing conversation memory...")
            
            history_context = "PREVIOUS CONVERSATION HISTORY:\n"
            # Loop through history excluding the latest prompt (to avoid duplication)
            for msg in st.session_state['message'][:-1]:
                role = "User" if msg["role"] == "human" else "AI"
                history_context += f"{role}: {msg['content']}\n"
            
            content_parts.append(history_context)
            time.sleep(0.5)

            # --- STEP 3: QUERY INTEGRATION ---
            # Combining the user's current question with the documents and history
            st.write("🔗 Contextualizing user query...")
            current_query = f"CURRENT USER QUESTION: {prompt_text}"
            content_parts.append(current_query)

            # --- STEP 4: TOKEN CALCULATION ---
            # Calculating the "cost" or weight of the request
            st.write("📊 Analyzing token usage...")
            count_resp = client.models.count_tokens(
                model="gemini-2.5-flash",
                contents=content_parts
            )
            token_count = count_resp.total_tokens
            st.info(f"ℹ️ Input Tokens: {token_count:,} tokens sent to model.")

            # --- STEP 5: GENERATION & STREAMING ---
            # Configuring the model behavior and safety rails
            st.write("⚡ Gemini 2.5 Flash is thinking...")
            
            # Optimized System Instruction for Document Analysis role
            system_instruction = f"""
            You are a highly capable Document Analysis AI.
            
            CORE INSTRUCTIONS:
            1. GROUNDING: Answer the user's question using ONLY the information found in the provided documents.
            2. CITATION: Always mention the source file name (e.g., [Source: file.pdf]) to back up your claims.
            3. LIMITATION: If the answer is NOT in the documents, explicitly state: "I cannot find that information in the provided documents." Do not hallucinate.
            4. TONE: Professional, technical, and concise.
            5. CONTEXT: Use the 'Conversation History' to understand follow-up questions.
            
            6. ***CRITICAL LANGUAGE RULE***: 
            The user explicitly selected to receive answers in: **{target_language}**.
            - **IGNORE** the language of the user's question.
            - **IGNORE** the language of the provided documents.
            - You MUST translate your thought process and final answer entirely into **{target_language}**.
            - Example: If the user asks in English but {target_language} is 'Indonesian', you MUST answer in Indonesian.
            """

            # Define Safety Settings to prevent the model from blocking valid content unnecessarily
            # We set threshold to BLOCK_ONLY_HIGH to make it less sensitive
            safety_settings = [
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_ONLY_HIGH"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_ONLY_HIGH"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_ONLY_HIGH"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH"
                ),
            ]

            generate_config = types.GenerateContentConfig(
                temperature = 0.3, # Low temperature for factual accuracy
                system_instruction = system_instruction,
                safety_settings = safety_settings
            )

            # Initiate the Streaming Request
            response_stream = client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=content_parts,
                config = generate_config
            )

            # Update status to success before rendering text
            status.update(label="✅ Insight Generated! Streaming response...", state="complete", expanded=False)
        
        # --- C. RENDER STREAMING RESPONSE ---
        with st.chat_message("ai"):
            placeholder = st.empty()
            full_text = ""

            # Loop through the stream chunks as they arrive
            for chunk in response_stream:
                try:
                    # Check if the chunk contains text (it might be empty if blocked by safety filter)
                    if chunk.text:
                        full_text += chunk.text
                        # Display text with a cursor effect ('▌')
                        placeholder.markdown(full_text + "▌") 
                except Exception:
                    # Handle Safety Filter blocks or Empty Chunks silently to avoid crashing
                    pass

            # Final render without the cursor
            placeholder.markdown(full_text)
            answer = full_text
                
    except Exception as e:
        # Error Handling: If anything goes wrong, display it clearly
        answer = f"⚠️ System Error: {e}"
        status.update(label="❌ Process Failed", state="error", expanded=False)
        st.error(answer)

    # Save AI Response to Session State History
    st.session_state["message"].append({"role": "ai", "content": answer})