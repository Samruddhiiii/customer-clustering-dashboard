import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# === Functions === #

def preprocess_data(filepath='Mall_Customers.csv'):
    df = pd.read_csv(filepath)
    df.rename(columns={'Genre': 'Gender'}, inplace=True)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    return df

def train_model(data, n_clusters=5):
    features = data[['Annual Income (k$)', 'Spending Score (1-100)']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(features)
    return data, kmeans

def plot_histograms(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df['Age'], kde=True, ax=axes[0], color='skyblue')
    sns.histplot(df['Annual Income (k$)'], kde=True, ax=axes[1], color='lightgreen')
    sns.histplot(df['Spending Score (1-100)'], kde=True, ax=axes[2], color='salmon')
    axes[0].set_title('Age Distribution')
    axes[1].set_title('Annual Income Distribution')
    axes[2].set_title('Spending Score Distribution')
    plt.tight_layout()
    return fig

def plot_heatmap(df):
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr(),
                cmap='magma_r', annot=True, linewidths=0.5)
    plt.title('Correlation Heatmap')
    return fig

def plot_violin_spending(df):
    fig = plt.figure(figsize=(10, 6))
    sns.violinplot(x='Gender', y='Spending Score (1-100)', data=df, inner=None)
    sns.swarmplot(x='Gender', y='Spending Score (1-100)', data=df, color='k', size=3)
    plt.title('Gender vs Spending Score')
    plt.xlabel('Gender (0=Female, 1=Male)')
    plt.ylabel('Spending Score (1-100)')
    return fig

def plot_clusters(data, y_kmeans, kmeans, xlabel, ylabel, n_clusters=5):
    colors = sns.color_palette("tab10", n_clusters)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(n_clusters):
        ax.scatter(data[y_kmeans == i, 0], data[y_kmeans == i, 1], s=100, c=[colors[i]], label=f'Cluster {i+1}')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               s=400, c='yellow', edgecolor='black', label='Centroid', marker='*')
    ax.set_title('Cluster Segmentation of Customers')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig

def elbow_method(features, max_clusters=10, title_suffix=''):
    wcss = []
    for i in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
        kmeans.fit(features)
        wcss.append(kmeans.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(1, max_clusters+1), wcss, marker='o')
    ax.set_title(f'The Elbow Method {title_suffix}')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('WCSS')
    return fig

# === Streamlit App === #

st.set_page_config(page_title="Mall Customers Segmentation", layout="wide")

st.title("🛍️ Mall Customers Segmentation")
st.write("This app performs KMeans clustering on mall customers and displays various analysis charts.")

@st.cache_data
def load_and_preprocess():
    return preprocess_data()

df = load_and_preprocess()

# Optional: Show raw data
if st.checkbox("📂 Show Raw Data"):
    st.subheader("Raw Dataset")
    st.dataframe(df)

# Data summary
st.subheader("📊 Data Summary")
st.write(df.describe())

# Gender distribution
st.subheader("👥 Gender Distribution")
genders = df['Gender'].value_counts().sort_index()
fig_gender, ax_gender = plt.subplots()
sns.barplot(x=genders.index, y=genders.values, palette=['pink', 'lightblue'], ax=ax_gender)
ax_gender.set_xticklabels(['Female (0)', 'Male (1)'])
ax_gender.set_ylabel('Count')
st.pyplot(fig_gender)

# Histograms
st.subheader("📈 Histograms of Age, Income, and Spending Score")
st.pyplot(plot_histograms(df))

# Violin plot
st.subheader("🎻 Gender vs Spending Score")
st.pyplot(plot_violin_spending(df))

# Elbow Method (always shown)
st.subheader("📉 Elbow Method to Determine Optimal Number of Clusters")
features_elbow = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
fig_elbow = elbow_method(features_elbow)
st.pyplot(fig_elbow)

# Cluster selection
k = st.slider("🔧 Select number of clusters for KMeans", 2, 10, 5)

# Train and display clusters
df, kmeans = train_model(df, n_clusters=k)

# Cluster plot: Income vs Spending
st.subheader("📍 Cluster Plot: Annual Income vs Spending Score")
features_1 = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
y_kmeans1 = df['Cluster'].values
fig_clusters1 = plot_clusters(features_1, y_kmeans1, kmeans,
                              'Annual Income (k$)', 'Spending Score (1-100)', n_clusters=k)
st.pyplot(fig_clusters1)

# Cluster plot: Age vs Spending
st.subheader("📍 Cluster Plot: Age vs Spending Score")
features_2 = df[['Age', 'Spending Score (1-100)']].values
kmeans2 = KMeans(n_clusters=k, random_state=42)
y_kmeans2 = kmeans2.fit_predict(features_2)
fig_clusters2 = plot_clusters(features_2, y_kmeans2, kmeans2, 'Age', 'Spending Score (1-100)', n_clusters=k)
st.pyplot(fig_clusters2)
