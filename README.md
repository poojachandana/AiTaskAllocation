Project_Link- https://aitaskallocation.streamlit.app/

# **🚀 AI-Powered Task Allocation System – Solution Overview**  

## **1️⃣ Solution Approach**  

This project is an **AI-based task allocation system** that **automatically assigns tasks to employees** based on their **skills, workload, and availability**. The system ensures that tasks are **fairly distributed** and employees are matched with the best-fitting tasks using **Natural Language Processing (NLP) and Machine Learning (ML)**.  

### **How It Works:**  
1. **Users Upload Data** 📂  
   - A **CSV file with tasks** (descriptions & required skills).  
   - A **CSV file with employees** (skills, availability, preferences).  

2. **AI Matches Tasks to Employees** 🤖  
   - Uses **BERT (a deep learning model)** to understand task descriptions & employee skills.  
   - **Cosine Similarity** is used to measure how well a task matches an employee’s skill set.  
   - Ensures **workload balancing** so no one is overloaded.  

3. **Displays AI-Generated Task Assignments** 📊  
   - A **table** lists who gets assigned which tasks.  
   - A **graph** visualizes match scores and workload distribution.  

4. **Allows Manual Adjustments** ✏  
   - Users can **reassign tasks** if needed.  
   - AI **suggests alternate employees** for each task based on match scores.  

5. **Stores & Syncs Data in Real-Time** 🔄  
   - Uses **Firebase** to store task assignments.  
   - Team members can use a **real-time chat feature** to discuss changes.  

6. **Users Can Download Final Task Assignments** 📥  
   - Generates a **CSV report** with all assigned tasks.  

---

## **2️⃣ Implementation Details – Technologies & Libraries Used**  

### **Key Technologies**  
✅ **Streamlit** → Creates the web-based user interface.  
✅ **Firebase Realtime Database** → Stores task allocations & enables real-time updates.  
✅ **NLP (BERT model)** → Understands and matches skills with tasks.  
✅ **Plotly** → Generates graphs for task allocation insights.  

### **Libraries Used**  
🔹 **pandas** → Reads and processes CSV files.  
🔹 **numpy** → Helps with numerical operations.  
🔹 **sentence-transformers** → Converts text data into embeddings for AI matching.  
🔹 **scikit-learn** → Provides the Cosine Similarity function for task matching.  
🔹 **streamlit-sortables** → Allows users to manually adjust task assignments.  
🔹 **firebase-admin** → Connects Streamlit to Firebase for real-time updates.  

---

## **3️⃣ Execution Steps – How to Run the Code?**  

### **Locally (on your own machine)**  
1️⃣ **Install Required Libraries**  
   ```bash
   pip install pandas numpy scikit-learn streamlit firebase-admin sentence-transformers plotly
   ```
2️⃣ **Set Up Firebase**  
   - Go to [Firebase Console](https://console.firebase.google.com/)  
   - Create a **Firebase Realtime Database**.  
   - Download the **Firebase private key JSON file**.  

3️⃣ **Run the Streamlit App**  
   ```bash
   streamlit run app.py
   ```
4️⃣ **Use the Web Interface**  
   - Upload CSV files for **tasks & employees**.  
   - View **AI-generated task allocations**.  
   - Make manual adjustments if needed.  
   - Save and **download final task assignments**.  

---

## **4️⃣ Dependencies – Required Software & APIs**  

✅ **Python 3.7+**  
✅ **Google Firebase** (for storing and updating data)  
✅ **Pretrained BERT Model** (`all-MiniLM-L6-v2`)  
✅ **Required Libraries**: pandas, numpy, scikit-learn, streamlit, firebase-admin, sentence-transformers, plotly  

---

## **5️⃣ Expected Output – What the System Will Deliver?**  

### **🎯 AI-Based Task Allocation**  
✔ **Tasks are assigned automatically** to the best-matching employees.  
✔ **Workload is balanced** so no one is overburdened.  
✔ **Match scores (%) are displayed** for each assignment.  

### **📊 Visual Reports**  
✔ A **table** shows task assignments.  
✔ A **bar graph** visualizes match scores and task distribution.  

### **🔄 Manual Adjustments & AI Suggestions**  
✔ Users can **reassign tasks manually**.  
✔ AI **suggests better matches** if needed.  

### **💬 Real-Time Chat System**  
✔ Team members can **chat & discuss assignments** in real-time.  

### **📥 Download Task Assignments**  
✔ Users can **download a CSV report** of the final task allocations.  

---

🚀 **Final Outcome:** A **smart, AI-driven, user-friendly task allocation system** that optimizes team productivity and reduces workload imbalances! Let me know if you want any modifications or additional features! 😊


