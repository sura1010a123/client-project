
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 1. Application configuration
# -------------------------------

st.set_page_config(
    page_title="OE Duplicate Checker",
    page_icon="🔎",
    layout="wide"
)

st.markdown("""
<style>
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background-color: #FBF5DF;
    }

    .stApp,
    .stApp p,
    .stApp label,
    .stApp h1,
    .stApp h2,
    .stApp h3 {
        color: #332F25;
    }

    .app-brand {
        margin: 0 0 0.25rem;
        color: #514630;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: 0.16em;
    }

    .app-title {
        margin: 0;
        color: #332F25;
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1.2;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input {
        background-color: #626262;
        color: #FFFFFF;
    }

    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {
        color: #EEEEEE;
        opacity: 1;
    }

    [data-testid="stTabs"] [role="tablist"] {
        gap: 0.75rem;
        border-bottom: none;
    }

    [data-testid="stTabs"] button[role="tab"] {
        background-color: #A87970;
        border: 1px solid #A87970;
        border-radius: 0.5rem;
        color: #FFFFFF;
        font-weight: 700;
        padding: 0.5rem 1rem;
    }

    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background-color: #916B63;
        border-color: #916B63;
        color: #FFFFFF;
    }

    [data-testid="stTabs"] button[role="tab"]:hover {
        background-color: #916B63;
        border-color: #916B63;
        color: #FFFFFF;
    }

    [data-testid="stButton"] button,
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #A87970;
        border-color: #A87970;
        color: #FFFFFF;
        font-weight: 700;
    }

    [data-testid="stButton"] button:hover,
    [data-testid="stFileUploader"] button:hover,
    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #916B63;
        border-color: #916B63;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

from pathlib import Path

DB = str(Path(__file__).resolve().parent / "oes.db")

# -------------------------------
# 2. Load the AI model
# -------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -------------------------------
# 3. Create the database
# -------------------------------

def get_connection():
    return sqlite3.connect(DB)

def create_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS oes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL,
                oe TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'used',
                team_member TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)

create_database()

# -------------------------------
# 4. Save an OE
# -------------------------------

def save_oe(project, oe, status, team_member):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO oes
            (project, oe, status, team_member, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            project.strip(),
            oe.strip(),
            status,
            team_member.strip(),
            datetime.now().isoformat(timespec="seconds")
        ))

# -------------------------------
# 5. Read OEs
# -------------------------------

def get_oes(project):
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM oes WHERE project = ?",
            conn,
            params=(project,)
        )

# -------------------------------
# 6. Find all similar OEs
# -------------------------------

def find_similar(new_oe, project, threshold):
    df = get_oes(project)

    if df.empty:
        return df

    # Create embeddings for existing responses
    old_embeddings = model.encode(
        df["oe"].tolist(),
        normalize_embeddings=True
    )

    # Create embedding for the new response
    new_embedding = model.encode(
        [new_oe],
        normalize_embeddings=True
    )

    # Compare all existing responses
    scores = cosine_similarity(
        new_embedding,
        old_embeddings
    )[0]

    df = df.copy()
    df["similarity"] = (scores * 100).round(2)

    # Show every match above the selected threshold
    matches = df[df["similarity"] >= threshold * 100]

    return matches.sort_values(
        "similarity",
        ascending=False
    )

# -------------------------------
# 7. Web interface
# -------------------------------

st.markdown(
    '<div class="app-brand">GENOSURV</div>'
    '<div class="app-title">🔎 OE Duplicate Checker</div>',
    unsafe_allow_html=True
)
st.caption("Find previously stored open-ended responses.")

tab1, tab2, tab3, tab4 = st.tabs([
    "Check OE",
    "Add OE",
    "Upload OEs",
    "Database"
])

# -------------------------------
# Tab 1: Check OE
# -------------------------------

with tab1:
    st.subheader("Check for similar responses")

    project = st.text_input(
        "Project ID",
        key="search_project",
        placeholder="e.g. PROJECT_1200"
    )

    new_oe = st.text_area(
        "Enter new OE",
        placeholder="Enter the respondent's answer here..."
    )

    threshold = st.slider(
        "Minimum similarity (%)",
        min_value=40,
        max_value=100,
        value=60,
        step=5
    )

    if st.button("Find similar OEs", type="primary"):
        if not project.strip() or not new_oe.strip():
            st.warning("Enter both a project ID and an OE.")
        else:
            with st.spinner("Comparing responses..."):
                matches = find_similar(
                    new_oe,
                    project.strip(),
                    threshold / 100
                )

            if matches.empty:
                st.success("No matches found above this threshold.")
            else:
                st.warning(
                    f"{len(matches)} similar OEs found."
                )

                display = matches[[
                    "similarity", "oe", "status",
                    "team_member", "created_at"
                ]].rename(columns={
                    "similarity": "Similarity (%)",
                    "oe": "Existing OE",
                    "status": "Status",
                    "team_member": "Team member",
                    "created_at": "Date added"
                })

                st.dataframe(
                    display,
                    use_container_width=True,
                    hide_index=True
                )

# -------------------------------
# Tab 2: Add one OE
# -------------------------------

with tab2:
    st.subheader("Add a response")

    with st.form("add_oe_form"):
        project = st.text_input("Project ID")
        oe = st.text_area("OE response")
        status = st.selectbox(
            "Status",
            ["used", "unused"]
        )
        member = st.text_input("Team member")

        submitted = st.form_submit_button("Save OE", type="primary")

        if submitted:
            if not project.strip() or not oe.strip():
                st.error("Project ID and OE are required.")
            else:
                save_oe(project, oe, status, member)
                st.success("OE saved successfully.")

# -------------------------------
# Tab 3: Bulk upload
# -------------------------------

with tab3:
    st.subheader("Upload multiple OEs")

    st.write(
        "Upload a CSV containing an 'oe' column. "
        "Optional columns: project, status, team_member."
    )

    uploaded = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        st.dataframe(data.head(10), use_container_width=True)

        if "oe" not in data.columns:
            st.error("Your CSV must contain an 'oe' column.")
        elif st.button("Import CSV"):
            imported = 0

            for _, row in data.iterrows():
                if pd.isna(row["oe"]) or not str(row["oe"]).strip():
                    continue

                project = str(row.get("project", "")).strip()
                status = str(row.get("status", "used")).strip()
                member = str(row.get("team_member", "")).strip()

                if not project or project.lower() == "nan":
                    st.error(
                        "Every row must have a project ID."
                    )
                    break

                if status not in ["used", "unused"]:
                    status = "used"

                save_oe(
                    project,
                    str(row["oe"]),
                    status,
                    member
                )
                imported += 1

            else:
                st.success(f"Imported {imported} OEs.")

# -------------------------------
# Tab 4: Browse database
# -------------------------------

with tab4:
    st.subheader("Stored responses")

    project = st.text_input(
        "Project ID to view",
        key="database_project"
    )

    if project.strip():
        data = get_oes(project.strip())

        st.metric("Stored OEs", len(data))

        if not data.empty:
            st.dataframe(
                data,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No OEs stored for this project.")