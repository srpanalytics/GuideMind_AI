import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# App config
st.set_page_config(page_title="AI Tool Hub", layout="wide")
st.title("🤖 GuideMind - AI Tool Hub")
st.markdown("Explore the best AI tools for every task — from content creation to automation.")

# 🌟 Modern Professional CSS
st.markdown("""
    <style>
    body, .stApp {
        background-color: #f4f6f9;
        font-family: 'Segoe UI', sans-serif;
    }

    .tool-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        min-height: 300px;
        border: 1px solid #e3e8ef;
    }

    .tool-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.08);
    }

    .tool-title {
        font-size: 22px;
        font-weight: 700;
        color: #1f3b76;
        margin-bottom: 6px;
    }

    .tool-category {
        font-size: 14px;
        font-weight: 500;
        color: #6c757d;
        margin-bottom: 12px;
    }

    .tool-description {
        font-size: 15px;
        color: #2e2e2e;
        line-height: 1.5;
        margin-bottom: 18px;
    }

    .button-link {
        background-color: #0056d2;
        color: white !important;
        padding: 10px 18px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 600;
        text-decoration: none;
        display: inline-block;
        margin-top: 12px;
        transition: all 0.2s ease;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.12);
    }

    .button-link:hover {
        background-color: #003b99;
        transform: scale(1.03);
        text-decoration: none;
    }

    .stButton > button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
        background-color: #e4e9f0;
        color: #333;
        border: none;
        padding: 8px 0;
        margin-top: 10px;
        transition: background 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #cdd7e3;
    }

    h2 {
        margin-top: 40px;
        color: #1c2c50;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #dee2e6;
    }

    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        border-radius: 8px;
        padding: 10px;
        font-size: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/tools.csv", encoding='latin1')
    return df

tools_df = load_data()

# Load NLP model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Bookmark state
if 'bookmarked' not in st.session_state:
    st.session_state['bookmarked'] = set()

# Sidebar filter
st.sidebar.subheader("Filter by Category")
categories = sorted(tools_df['Category'].dropna().unique())
selected_category = st.sidebar.selectbox("Choose a category", ["All"] + categories)

# Search bar
query = st.text_input("🔍 What do you want to do?", placeholder="e.g. generate blog, write code...")

# NLP Search
if query:
    tool_embeddings = model.encode(tools_df["Description"].fillna("").astype(str), show_progress_bar=False)
    query_embedding = model.encode([query])[0].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, tool_embeddings)[0]
    tools_df["Similarity"] = similarities
    threshold = 0.4
    filtered_df = tools_df[tools_df["Similarity"] >= threshold].sort_values("Similarity", ascending=False)
    if filtered_df.empty:st.warning("No tools found matching your query. Try a different search term.")
else:
    filtered_df = tools_df.copy()

# Apply category filter
if selected_category != "All":
    filtered_df = filtered_df[filtered_df["Category"] == selected_category]

# Results
st.markdown("### Matching AI Tools")

cols = st.columns(3)
for i, (_, row) in enumerate(filtered_df.iterrows()):
    with cols[i % 3]:
        with st.container():
            st.markdown(f"""
                <div class="tool-card">
                    <div class="tool-title">{row['Tool Name']}</div>
                    <div class="tool-category">📁 {row['Category']}</div>
                    <div class="tool-description">{row['Description'][:160]}...</div>
                    <a class="button-link" href="{row['URL']}" target="_blank">🌐 Visit Tool</a>
            """, unsafe_allow_html=True)

            tool_key = row["Tool Name"]
            if tool_key not in st.session_state['bookmarked']:
                if st.button("⭐ Save", key=f"save_{i}"):
                    st.session_state['bookmarked'].add(tool_key)
            else:
                if st.button("✅ Saved", key=f"saved_{i}"):
                    st.session_state['bookmarked'].remove(tool_key)

            st.markdown("</div>", unsafe_allow_html=True)

# Saved tools
if st.session_state['bookmarked']:
    st.markdown("## ⭐ Saved Tools")
    for tool in st.session_state['bookmarked']:
        tool_row = tools_df[tools_df["Tool Name"] == tool].iloc[0]
        st.write(f"🔗 [{tool_row['Tool Name']}]({tool_row['URL']}) — {tool_row['Category']}")
