import streamlit as st
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
import firebase_admin
from firebase_admin import credentials, db
import re
import json

# -------------------------------
# Firebase Setup using Streamlit Secrets
# -------------------------------
if not firebase_admin._apps:
    firebase_key = {
        "type": st.secrets["firebase"]["type"],
        "project_id": st.secrets["firebase"]["project_id"],
        "private_key_id": st.secrets["firebase"]["private_key_id"],
        "private_key": st.secrets["firebase"]["private_key"].replace("\\n", "\n"),
        "client_email": st.secrets["firebase"]["client_email"],
        "client_id": st.secrets["firebase"]["client_id"],
        "auth_uri": st.secrets["firebase"]["auth_uri"],
        "token_uri": st.secrets["firebase"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"],
        "universe_domain": st.secrets["firebase"]["universe_domain"]
    }

    cred = credentials.Certificate(firebase_key)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://ai-task-allocation-default-rtdb.firebaseio.com/'
    })

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AI Task Allocation", layout="wide")
st.title("🔹 AI Agent for Smart Task Allocation")
st.markdown("### 📌 Upload CSV files to allocate tasks based on skills and workload.")

# Sidebar
st.sidebar.header("⚙ Settings")
min_match_score = st.sidebar.slider("🔍 Minimum Match Score (%)", 0, 100, 50)
max_tasks_per_person = st.sidebar.slider("📊 Max Tasks per Person", 1, 10, 3)

# Upload
st.subheader("📂 Upload Your Data")
task_file = st.file_uploader("Upload Tasks CSV", type="csv")
individual_file = st.file_uploader("Upload Individuals CSV", type="csv")

# Load data
if task_file:
    tasks = pd.read_csv(task_file)
else:
    try:
        tasks = pd.read_csv("data/tasks.csv")
        st.info("✅ Using default tasks.csv from /data")
    except FileNotFoundError:
        st.error("❌ tasks.csv not found. Please upload the file.")
        st.stop()

if individual_file:
    individuals = pd.read_csv(individual_file)
else:
    try:
        individuals = pd.read_csv("data/individuals.csv")
        st.info("✅ Using default individuals.csv from /data")
    except FileNotFoundError:
        st.error("❌ individuals.csv not found. Please upload the file.")
        st.stop()

# Normalize
individuals.columns = individuals.columns.str.strip().str.lower()
tasks.columns = tasks.columns.str.strip().str.lower()

if "skills" not in individuals.columns:
    st.error("Error: 'Skills' column not found in Individuals CSV.")
    st.stop()

# Show Data
st.write("### 👥 Individuals Dataset")
st.dataframe(individuals.head())
st.write("### 📋 Tasks Dataset")
st.dataframe(tasks.head())

# Load Hugging Face model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

individual_embeddings = model.encode(individuals['skills'].astype(str).tolist(), convert_to_tensor=True)
task_embeddings = model.encode(tasks['required_skills'].astype(str).tolist(), convert_to_tensor=True)
cosine_scores = util.pytorch_cos_sim(task_embeddings, individual_embeddings)

# Extract numeric availability
def extract_numeric_availability(value):
    numbers = re.findall(r'\d+', str(value))
    return int(numbers[0]) if numbers else max_tasks_per_person

# Allocation Function
def allocate_with_balanced_workload(cosine_scores, individuals, tasks):
    allocations = []
    suggestions = []
    workload = {name: 0 for name in individuals["name"]}
    availability = {
        name: extract_numeric_availability(individuals[individuals["name"] == name]["availability"].values[0])
        for name in individuals["name"]
    }

    for task_idx, task in enumerate(tasks.itertuples()):
        sorted_indices = cosine_scores[task_idx].argsort(descending=True)
        best_match_idx = None

        for idx in sorted_indices.tolist():
            candidate = individuals.iloc[idx]
            if workload[candidate["name"]] < min(max_tasks_per_person, availability[candidate["name"]]):
                best_match_idx = idx
                break

        if best_match_idx is None:
            best_match_idx = sorted_indices[0].item()

        best_match_score = float(cosine_scores[task_idx][best_match_idx].item() * 100)
        selected_individual = individuals.iloc[best_match_idx]
        workload[selected_individual["name"]] += 1

        task_data = {
            "Task": task.task_description,
            "Assigned To": selected_individual["name"],
            "Skills Matched": selected_individual["skills"],
            "Availability": selected_individual["availability"],
            "Preference": selected_individual.get("preferences", "N/A"),
            "Match Score (%)": round(best_match_score, 2),
            "Status": "To Do"
        }
        allocations.append(task_data)

        top_suggestions = [{
            "Candidate": individuals.iloc[idx]["name"],
            "Match Score (%)": round(float(cosine_scores[task_idx][idx].item() * 100), 2)
        } for idx in sorted_indices[:3].tolist()]
        suggestions.append(top_suggestions)

    return pd.DataFrame(allocations), suggestions

# Run Allocation
allocation_results, suggestions = allocate_with_balanced_workload(cosine_scores, individuals, tasks)
st.write("### 📊 AI-Based Task Allocation")
st.dataframe(allocation_results)

# Plot
fig = px.bar(allocation_results, x="Task", y="Match Score (%)", color="Assigned To", title="Match Score per Task")
st.plotly_chart(fig)

# Manual Reassign
st.write("### 🔄 Manual Task Reassignment & AI Suggestions")
updated_allocations = allocation_results.copy()

for i in range(len(updated_allocations)):
    st.write(f"#### Task: {updated_allocations.iloc[i]['Task']}")
    current_assignee = updated_allocations.iloc[i]["Assigned To"]
    index = int(individuals[individuals["name"] == current_assignee].index[0])

    new_assignee = st.selectbox(
        f"Reassign Task {i+1}",
        individuals["name"].tolist(),
        index=index
    )
    updated_allocations.at[i, "Assigned To"] = new_assignee

    st.write("💡 AI Suggestions:")
    for suggestion in suggestions[i]:
        st.write(f"➡ {suggestion['Candidate']} (Match: {suggestion['Match Score (%)']}%)")

# Save Button
if st.button("✅ Save Changes"):
    db.reference("/task_allocations").set(updated_allocations.to_dict(orient="records"))
    st.write("### ✅ Updated Task Allocations (Synced in Real-Time)")
    st.dataframe(updated_allocations)
    fig_updated = px.bar(updated_allocations, x="Task", y="Match Score (%)", color="Assigned To", title="Updated Match Score per Task")
    st.plotly_chart(fig_updated)

# Optional Chat
st.write("### 💬 Team Chat System")
chat_ref = db.reference("/chat")
chat_messages = chat_ref.get()

if chat_messages:
    for msg in chat_messages.values():
        st.write(f"{msg['user']}: {msg['message']}")

new_message = st.text_input("Type a message:")
if st.button("Send") and new_message:
    chat_ref.push({"user": "User", "message": new_message})
    st.rerun()

# Download Button
csv = updated_allocations.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Allocation Results", data=csv, file_name="allocations.csv", mime="text/csv")
