# Mall Customers Segmentation (KMeans + Streamlit)

This project applies **KMeans clustering** to segment mall customers based on key features like Age, Annual Income, and Spending Score. It includes an **interactive dashboard built with Streamlit** for real-time cluster visualization.

---

## Key Features

- 💡 **KMeans Clustering** to segment customers into behavior-based groups
- 📊 **Visualizations**: Cluster plots, heatmaps, histograms, violin plots, and the Elbow method
- 🖥️ **Interactive Streamlit App** for real-time exploration of clustering results
- 📈 Two clustering perspectives:
  - **Annual Income vs Spending Score**
  - **Age vs Spending Score**
- 📍 Optimal number of clusters selected using the **Elbow Method**

---

## 🚀 Why Streamlit?

This project was initially built using Flask, but it was **time-consuming and required manual integration of visualizations**. By switching to **Streamlit**, development became much more efficient:

- No need for custom HTML templates
- Visualizations directly integrated with minimal boilerplate
- Real-time updates with sliders and checkboxes

> 🔄 Streamlit allowed rapid prototyping, reduced complexity, and made it easier to focus on **data insights** rather than frontend setup.

---

## 📁 Files Included

- `Mall_Customers.csv`: Customer dataset with Age, Gender, Income, and Spending Score
- `mall_streamlit_app.py`: Streamlit-based frontend for data visualization
- `two.py`: Python script with all clustering logic and helper functions

---
