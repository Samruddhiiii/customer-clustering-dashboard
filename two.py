
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np

def preprocess_data(filepath='Mall_Customers.csv'):
    df = pd.read_csv(filepath)
    df.rename(columns={'Genre': 'Gender'}, inplace=True)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    return df

def train_kmeans(features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++')
    labels = kmeans.fit_predict(features)
    return kmeans, labels

def plot_histograms(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df['Age'], kde=True, ax=axes[0])
    sns.histplot(df['Annual Income (k$)'], kde=True, ax=axes[1])
    sns.histplot(df['Spending Score (1-100)'], kde=True, ax=axes[2])
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

def plot_violin_income(df):
    fig = plt.figure(figsize=(10, 6))
    sns.violinplot(x='Gender', y='Annual Income (k$)', data=df, inner=None)
    sns.swarmplot(x='Gender', y='Annual Income (k$)', data=df, color='k', size=3)
    plt.title('Gender vs Annual Income')
    plt.xlabel('Gender (0=Female, 1=Male)')
    plt.ylabel('Annual Income (k$)')
    return fig

def plot_violin_spending(df):
    fig = plt.figure(figsize=(10, 6))
    sns.violinplot(x='Gender', y='Spending Score (1-100)', data=df, inner=None)
    sns.swarmplot(x='Gender', y='Spending Score (1-100)', data=df, color='k', size=3)
    plt.title('Gender vs Spending Score')
    plt.xlabel('Gender (0=Female, 1=Male)')
    plt.ylabel('Spending Score (1-100)')
    return fig

def plot_pairplot(df):
    sns.set(style="ticks")
    pairplot_fig = sns.pairplot(df[['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)']])
    return pairplot_fig

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

def plot_clusters(data, labels, kmeans, xlabel, ylabel, n_clusters=5):
    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink', 'gray']
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(n_clusters):
        ax.scatter(data[labels == i, 0], data[labels == i, 1], s=100, c=colors[i], label=f'Cluster {i+1}')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=400, c='yellow', marker='*', label='Centroid')
    ax.set_title('Cluster Segmentation of Customers')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig

def plot_silhouette(features, labels, n_clusters, title_suffix=''):
    silhouette_avg = silhouette_score(features, labels)
    sample_silhouette_values = silhouette_samples(features, labels)
    fig, ax = plt.subplots(figsize=(10,6))
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1))
        y_lower = y_upper + 10
    ax.set_title(f'Silhouette Plot {title_suffix} (avg={silhouette_avg:.2f})')
    ax.set_xlabel('Silhouette coefficient values')
    ax.set_ylabel('Cluster label')
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1])
    return fig

def cluster_profile(df, labels):
    df['Cluster'] = labels
    profile = df.groupby('Cluster').mean()[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    return profile

if __name__ == "__main__":
    d_customer = preprocess_data('Mall_Customers.csv')
    print(d_customer.head())
    
    # Basic info (can comment out in Streamlit)
    print(d_customer.tail())
    print(d_customer.columns)
    print(d_customer.shape)
    print(d_customer.describe())
    print(d_customer.isnull().sum())
    d_customer.info()

    # Gender count barplot
    genders = d_customer.Gender.value_counts()
    sns.set_style("darkgrid")
    plt.figure(figsize=(10,4))
    sns.barplot(x=genders.index, y=genders.values)
    plt.xlabel('Gender (0=Female, 1=Male)')
    plt.ylabel('Count')
    plt.title('Gender Distribution')
    plt.show()

    # Histograms
    fig_hist = plot_histograms(d_customer)
    fig_hist.show()

    # Pairplot
    pairplot_fig = plot_pairplot(d_customer)
    plt.show()

    # Heatmap
    fig_heatmap = plot_heatmap(d_customer)
    fig_heatmap.show()

    # Violin plots
    fig_violin_income = plot_violin_income(d_customer)
    fig_violin_income.show()
    fig_violin_spending = plot_violin_spending(d_customer)
    fig_violin_spending.show()

    # Boxplot Gender vs Age
    sns.catplot(x='Gender', y='Age', kind='box', data=d_customer)
    plt.xlabel('Gender (0=Female, 1=Male)')
    plt.title('Gender vs Age')
    plt.show()

    # ========== Clustering 1: Annual Income & Spending Score ==========
    features_1 = d_customer[['Annual Income (k$)', 'Spending Score (1-100)']].values
    fig_elbow1 = elbow_method(features_1, max_clusters=10, title_suffix='(Income & Spending)')
    fig_elbow1.show()

    kmeans1, labels1 = train_kmeans(d_customer, n_clusters=5)
    fig_clusters1 = plot_clusters(features_1, labels1, kmeans1, 'Annual Income (k$)', 'Spending Score (1-100)', n_clusters=5)
    fig_clusters1.show()

    fig_silhouette1 = plot_silhouette(features_1, labels1, n_clusters=5, title_suffix='(Income & Spending)')
    fig_silhouette1.show()

    print("Cluster Profiles (Income & Spending):")
    print(cluster_profile(d_customer[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], labels1))

    # ========== Clustering 2: Age & Spending Score ==========
    features_2 = d_customer[['Age', 'Spending Score (1-100)']].values
    fig_elbow2 = elbow_method(features_2, max_clusters=10, title_suffix='(Age & Spending)')
    fig_elbow2.show()

    kmeans2, labels2 = train_kmeans(pd.DataFrame(features_2, columns=['Age', 'Spending Score']), n_clusters=5)
    fig_clusters2 = plot_clusters(features_2, labels2, kmeans2, 'Age', 'Spending Score (1-100)', n_clusters=5)
    fig_clusters2.show()

    fig_silhouette2 = plot_silhouette(features_2, labels2, n_clusters=5, title_suffix='(Age & Spending)')
    fig_silhouette2.show()

    print("Cluster Profiles (Age & Spending):")
    print(cluster_profile(d_customer[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], labels2))

