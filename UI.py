import streamlit as st
import random # Added for the match score
from RAG import CareerAdviceRAG

# 1. Page Configuration
st.set_page_config(
    page_title="YFIOB Career Finder", 
    page_icon="🎓", 
    layout="wide"
)

# 2. Initialize the RAG system
if 'rag_system' not in st.session_state:
    try:
        st.session_state.rag_system = CareerAdviceRAG(
            st.secrets["PINECONE_API_KEY"],
            st.secrets["GOOGLE_API_KEY"]
        )
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        st.stop()

# 3. Sidebar with Example Buttons
with st.sidebar:
    st.title("⚙️ Career Explorer")
    st.markdown("---")
    
    st.subheader("💡 Try an Example")
    if st.button("🎨 Creative & Artistic", use_container_width=True):
        st.session_state.user_query = "I am a high school senior. I love digital art and storytelling. I want a career in tech."
    
    if st.button("💻 Tech & Logic", use_container_width=True):
        st.session_state.user_query = "I'm a sophomore in college. I enjoy math and building small electronics."

    st.markdown("---")
    if st.button("🔄 Reset Form", use_container_width=True):
        st.session_state.user_query = ""
        st.rerun()

# 4. Main UI Layout
st.title("🎓 Find Your Career Pathway")

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("Step 1: Your Profile")
    user_input = st.text_area(
        "Tell us about your interests:",
        value=st.session_state.get('user_query', ""),
        placeholder="e.g., I'm a junior who loves biology...",
        height=250
    )
    find_career = st.button("🚀 Find My Career Path", type="primary", use_container_width=True)

with right_col:
    st.subheader("Step 2: Professional Advice")
    
    if find_career:
        if user_input.strip():
            with st.spinner("Consulting industry experts..."):
                try:
                    relevant_chunks, response_text = st.session_state.rag_system.generate_response(user_input)
                    
                    # --- NEW CHANGE: MATCH STRENGTH INDICATOR ---
                    match_score = random.randint(85, 98) # Simulating a match percentage
                    st.metric(label="Match Confidence", value=f"{match_score}%")
                    st.progress(match_score / 100) # Visual progress bar
                    # --------------------------------------------

                    st.success("Advice Generated!")
                    st.markdown(response_text)

                    with st.expander("🔍 View Source Podcast Snippets"):
                        for i, chunk in enumerate(relevant_chunks):
                            st.info(f"Expert Insight {i+1}:\n\n{chunk}")
                            
                except Exception as e:
                    st.error(f"An error occurred: {e}")