import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

def generate_sample_data(n_customers=1000):
    """Generate sample retail transaction data for demonstration"""
    np.random.seed(42)
    
    # Generate customer IDs
    customer_ids = [f"CUST_{i:04d}" for i in range(1, n_customers + 1)]
    
    # Generate transaction data
    data = []
    base_date = datetime(2023, 1, 1)
    
    for customer_id in customer_ids:
        # Random number of transactions per customer (1-20)
        n_transactions = np.random.poisson(5) + 1
        
        for _ in range(n_transactions):
            # Random transaction date within last 2 years
            days_ago = np.random.randint(0, 730)
            transaction_date = base_date + timedelta(days=days_ago)
            
            # Random monetary value (skewed distribution)
            monetary = np.random.lognormal(mean=3, sigma=1)
            
            data.append({
                'customer_id': customer_id,
                'transaction_date': transaction_date,
                'monetary_value': round(monetary, 2)
            })
    
    return pd.DataFrame(data)

def calculate_rfm(df):
    """Calculate RFM metrics from transaction data"""
    # Get the most recent date in the dataset
    current_date = df['transaction_date'].max() + timedelta(days=1)
    
    # Calculate RFM metrics
    rfm = df.groupby('customer_id').agg({
        'transaction_date': lambda x: (current_date - x.max()).days,  # Recency
        'customer_id': 'count',  # Frequency
        'monetary_value': 'sum'  # Monetary
    }).reset_index()
    
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    return rfm

def perform_kmeans_clustering(rfm_data, n_clusters, features_to_use):
    """Perform K-means clustering on RFM data"""
    # Select features for clustering
    X = rfm_data[features_to_use].copy()
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, clusters)
    
    # Add cluster labels to the original data
    rfm_clustered = rfm_data.copy()
    rfm_clustered['cluster'] = clusters
    
    return rfm_clustered, kmeans, scaler, silhouette_avg, X_scaled

def find_optimal_clusters(rfm_data, features_to_use, max_clusters=10):
    """Find optimal number of clusters using elbow method and silhouette score"""
    X = rfm_data[features_to_use].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, clusters))
    
    return cluster_range, inertias, silhouette_scores

def create_cluster_summary(rfm_clustered):
    """Create summary statistics for each cluster"""
    summary = rfm_clustered.groupby('cluster').agg({
        'recency': ['mean', 'median'],
        'frequency': ['mean', 'median'],
        'monetary': ['mean', 'median'],
        'customer_id': 'count'
    }).round(2)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()
    
    return summary

def main():
    # Main header
    st.markdown('<div class="main-header">🛍️ Customer Segmentation Dashboard</div>', unsafe_allow_html=True)
    st.markdown("### Analyze customer behavior using RFM analysis and K-means clustering")
    
    # Sidebar for controls
    st.sidebar.header("🎛️ Configuration Panel")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["Sample Data", "Upload Your Data"]
    )
    
    # Initialize session state
    if 'rfm_data' not in st.session_state:
        st.session_state.rfm_data = None
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    
    # Data loading section
    if data_source == "Sample Data":
        st.sidebar.info("Using generated sample data for demonstration")
        n_customers = st.sidebar.slider("Number of customers:", 100, 2000, 1000, 100)
        
        if st.sidebar.button("Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                st.session_state.raw_data = generate_sample_data(n_customers)
                st.session_state.rfm_data = calculate_rfm(st.session_state.raw_data)
            st.sidebar.success("Sample data generated!")
    
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload your cleaned data (CSV)",
            type=['csv'],
            help="Expected columns: customer_id, transaction_date, monetary_value"
        )
        
        if uploaded_file is not None:
            try:
                st.session_state.raw_data = pd.read_csv(uploaded_file)
                st.session_state.raw_data['transaction_date'] = pd.to_datetime(st.session_state.raw_data['transaction_date'])
                st.session_state.rfm_data = calculate_rfm(st.session_state.raw_data)
                st.sidebar.success("Data uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading data: {e}")
    
    # Main content
    if st.session_state.rfm_data is not None:
        # Display data overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", len(st.session_state.rfm_data))
        with col2:
            st.metric("Avg Recency (days)", f"{st.session_state.rfm_data['recency'].mean():.1f}")
        with col3:
            st.metric("Avg Frequency", f"{st.session_state.rfm_data['frequency'].mean():.1f}")
        with col4:
            st.metric("Avg Monetary ($)", f"{st.session_state.rfm_data['monetary'].mean():.2f}")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📊 RFM Analysis", "🎯 K-means Clustering", "📈 Cluster Analysis", "💡 Business Insights"])
        
        with tab1:
            st.markdown('<div class="sub-header">RFM Distribution Analysis</div>', unsafe_allow_html=True)
            
            # RFM histograms
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Recency Distribution', 'Frequency Distribution', 
                                  'Monetary Distribution', 'RFM Correlation'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"type": "heatmap"}]]
                )
                
                # Recency histogram
                fig_hist.add_trace(
                    go.Histogram(x=st.session_state.rfm_data['recency'], name='Recency', 
                               marker_color='lightblue', opacity=0.7),
                    row=1, col=1
                )
                
                # Frequency histogram
                fig_hist.add_trace(
                    go.Histogram(x=st.session_state.rfm_data['frequency'], name='Frequency', 
                               marker_color='lightgreen', opacity=0.7),
                    row=1, col=2
                )
                
                # Monetary histogram
                fig_hist.add_trace(
                    go.Histogram(x=st.session_state.rfm_data['monetary'], name='Monetary', 
                               marker_color='lightcoral', opacity=0.7),
                    row=2, col=1
                )
                
                # Correlation heatmap
                corr_matrix = st.session_state.rfm_data[['recency', 'frequency', 'monetary']].corr()
                fig_hist.add_trace(
                    go.Heatmap(z=corr_matrix.values, 
                             x=corr_matrix.columns, 
                             y=corr_matrix.columns,
                             colorscale='RdBu',
                             text=corr_matrix.round(2).values,
                             texttemplate="%{text}",
                             textfont={"size": 12}),
                    row=2, col=2
                )
                
                fig_hist.update_layout(height=600, showlegend=False, title_text="RFM Metrics Overview")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # RFM 3D scatter plot
                fig_3d = px.scatter_3d(
                    st.session_state.rfm_data, 
                    x='recency', y='frequency', z='monetary',
                    title='3D RFM Scatter Plot',
                    labels={'recency': 'Recency (days)', 'frequency': 'Frequency', 'monetary': 'Monetary ($)'},
                    opacity=0.6
                )
                fig_3d.update_layout(height=600)
                st.plotly_chart(fig_3d, use_container_width=True)
        
        with tab2:
            st.markdown('<div class="sub-header">K-means Clustering Configuration</div>', unsafe_allow_html=True)
            
            # Clustering parameters
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Clustering Parameters")
                
                # Feature selection
                st.write("**Select features for clustering:**")
                use_recency = st.checkbox("Recency", value=True)
                use_frequency = st.checkbox("Frequency", value=True)
                use_monetary = st.checkbox("Monetary", value=True)
                
                features_to_use = []
                if use_recency:
                    features_to_use.append('recency')
                if use_frequency:
                    features_to_use.append('frequency')
                if use_monetary:
                    features_to_use.append('monetary')
                
                if len(features_to_use) == 0:
                    st.error("Please select at least one feature for clustering!")
                    return
                
                # Number of clusters
                n_clusters = st.slider("Number of clusters:", 2, 10, 4)
                
                # Find optimal clusters button
                if st.button("🔍 Find Optimal Clusters"):
                    with st.spinner("Analyzing optimal cluster count..."):
                        cluster_range, inertias, silhouette_scores = find_optimal_clusters(
                            st.session_state.rfm_data, features_to_use
                        )
                        
                        # Plot elbow curve and silhouette scores
                        fig_optimal = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=('Elbow Method', 'Silhouette Score'),
                            x_title='Number of Clusters'
                        )
                        
                        fig_optimal.add_trace(
                            go.Scatter(x=list(cluster_range), y=inertias, mode='lines+markers', 
                                     name='Inertia', line=dict(color='blue')),
                            row=1, col=1
                        )
                        
                        fig_optimal.add_trace(
                            go.Scatter(x=list(cluster_range), y=silhouette_scores, mode='lines+markers', 
                                     name='Silhouette Score', line=dict(color='green')),
                            row=1, col=2
                        )
                        
                        fig_optimal.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig_optimal, use_container_width=True)
                        
                        # Recommend optimal clusters
                        optimal_idx = np.argmax(silhouette_scores)
                        optimal_clusters = list(cluster_range)[optimal_idx]
                        st.info(f"💡 Recommended number of clusters: **{optimal_clusters}** (highest silhouette score)")
                
                # Perform clustering button
                if st.button("🎯 Perform Clustering", type="primary"):
                    with st.spinner("Performing K-means clustering..."):
                        rfm_clustered, kmeans_model, scaler, silhouette_avg, X_scaled = perform_kmeans_clustering(
                            st.session_state.rfm_data, n_clusters, features_to_use
                        )
                        
                        # Store results in session state
                        st.session_state.rfm_clustered = rfm_clustered
                        st.session_state.kmeans_model = kmeans_model
                        st.session_state.scaler = scaler
                        st.session_state.features_used = features_to_use
                        st.session_state.n_clusters = n_clusters
                        
                    st.success(f"✅ Clustering completed! Silhouette Score: {silhouette_avg:.3f}")
            
            with col2:
                if 'rfm_clustered' in st.session_state:
                    st.subheader("Clustering Results")
                    
                    # Cluster visualization
                    if len(features_to_use) >= 2:
                        # 2D scatter plot
                        fig_clusters = px.scatter(
                            st.session_state.rfm_clustered,
                            x=features_to_use[0],
                            y=features_to_use[1],
                            color='cluster',
                            title=f'Customer Clusters ({features_to_use[0]} vs {features_to_use[1]})',
                            color_continuous_scale='viridis'
                        )
                        fig_clusters.update_layout(height=400)
                        st.plotly_chart(fig_clusters, use_container_width=True)
                        
                        # If 3 features, show 3D plot
                        if len(features_to_use) == 3:
                            fig_3d_clusters = px.scatter_3d(
                                st.session_state.rfm_clustered,
                                x=features_to_use[0],
                                y=features_to_use[1],
                                z=features_to_use[2],
                                color='cluster',
                                title='3D Customer Clusters',
                                color_continuous_scale='viridis'
                            )
                            fig_3d_clusters.update_layout(height=500)
                            st.plotly_chart(fig_3d_clusters, use_container_width=True)
        
        with tab3:
            if 'rfm_clustered' in st.session_state:
                st.markdown('<div class="sub-header">Detailed Cluster Analysis</div>', unsafe_allow_html=True)
                
                # Cluster summary statistics
                cluster_summary = create_cluster_summary(st.session_state.rfm_clustered)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📋 Cluster Summary Statistics")
                    st.dataframe(cluster_summary, use_container_width=True)
                
                with col2:
                    st.subheader("📊 Cluster Size Distribution")
                    cluster_counts = st.session_state.rfm_clustered['cluster'].value_counts().sort_index()
                    
                    fig_pie = px.pie(
                        values=cluster_counts.values,
                        names=[f'Cluster {i}' for i in cluster_counts.index],
                        title='Customer Distribution Across Clusters'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Cluster comparison charts
                st.subheader("📈 Cluster Comparison")
                
                # Box plots for each RFM metric
                metrics = ['recency', 'frequency', 'monetary']
                fig_box = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=[f'{metric.title()} by Cluster' for metric in metrics]
                )
                
                for i, metric in enumerate(metrics):
                    for cluster in sorted(st.session_state.rfm_clustered['cluster'].unique()):
                        cluster_data = st.session_state.rfm_clustered[
                            st.session_state.rfm_clustered['cluster'] == cluster
                        ][metric]
                        
                        fig_box.add_trace(
                            go.Box(y=cluster_data, name=f'Cluster {cluster}', 
                                 showlegend=i==0),
                            row=1, col=i+1
                        )
                
                fig_box.update_layout(height=400, title_text="RFM Metrics Distribution by Cluster")
                st.plotly_chart(fig_box, use_container_width=True)
                
                # Interactive cluster explorer
                st.subheader("🔍 Interactive Cluster Explorer")
                selected_cluster = st.selectbox(
                    "Select a cluster to explore:",
                    options=sorted(st.session_state.rfm_clustered['cluster'].unique()),
                    format_func=lambda x: f"Cluster {x}"
                )
                
                cluster_data = st.session_state.rfm_clustered[
                    st.session_state.rfm_clustered['cluster'] == selected_cluster
                ]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Customers in Cluster", len(cluster_data))
                with col2:
                    st.metric("Avg Recency", f"{cluster_data['recency'].mean():.1f} days")
                with col3:
                    st.metric("Avg Monetary", f"${cluster_data['monetary'].mean():.2f}")
                
                # Show sample customers from selected cluster
                st.write(f"**Sample customers from Cluster {selected_cluster}:**")
                st.dataframe(cluster_data.head(10), use_container_width=True)
            
            else:
                st.info("👆 Please perform clustering first in the K-means Clustering tab!")
        
        with tab4:
            if 'rfm_clustered' in st.session_state:
                st.markdown('<div class="sub-header">Business Insights & Recommendations</div>', unsafe_allow_html=True)
                
                # Generate cluster insights
                cluster_summary = create_cluster_summary(st.session_state.rfm_clustered)
                
                st.subheader("🎯 Customer Segment Profiles")
                
                for cluster_id in sorted(st.session_state.rfm_clustered['cluster'].unique()):
                    cluster_info = cluster_summary[cluster_summary['cluster'] == cluster_id].iloc[0]
                    
                    with st.expander(f"Cluster {cluster_id} - Analysis & Strategy"):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.write("**Key Metrics:**")
                            st.write(f"• Customers: {int(cluster_info['customer_id_count'])}")
                            st.write(f"• Avg Recency: {cluster_info['recency_mean']:.1f} days")
                            st.write(f"• Avg Frequency: {cluster_info['frequency_mean']:.1f}")
                            st.write(f"• Avg Monetary: ${cluster_info['monetary_mean']:.2f}")
                        
                        with col2:
                            # Generate insights based on RFM values
                            recency = cluster_info['recency_mean']
                            frequency = cluster_info['frequency_mean']
                            monetary = cluster_info['monetary_mean']
                            
                            # Simple segmentation logic
                            if recency < 30 and frequency > 5 and monetary > 500:
                                segment_name = "💎 VIP Customers"
                                insights = [
                                    "These are your best customers - high value, frequent buyers",
                                    "Focus on retention and loyalty programs",
                                    "Offer premium products and exclusive deals",
                                    "Gather feedback to improve experience"
                                ]
                            elif recency < 60 and frequency > 3:
                                segment_name = "⭐ Regular Customers"
                                insights = [
                                    "Consistent, reliable customer base",
                                    "Cross-sell and upsell opportunities",
                                    "Regular communication and newsletters",
                                    "Reward consistency with loyalty points"
                                ]
                            elif recency > 180:
                                segment_name = "😴 Dormant Customers"
                                insights = [
                                    "Haven't purchased recently - risk of churn",
                                    "Re-engagement campaigns needed",
                                    "Special offers to win them back",
                                    "Survey to understand why they stopped buying"
                                ]
                            elif frequency == 1:
                                segment_name = "🆕 One-time Buyers"
                                insights = [
                                    "Made only one purchase - conversion opportunity",
                                    "Follow-up campaigns to encourage repeat purchase",
                                    "Onboarding sequence for new customers",
                                    "Gather feedback on first purchase experience"
                                ]
                            else:
                                segment_name = "📊 Standard Customers"
                                insights = [
                                    "Average engagement and value",
                                    "Potential for growth with right strategy",
                                    "A/B test different approaches",
                                    "Monitor for changes in behavior"
                                ]
                            
                            st.write(f"**Segment: {segment_name}**")
                            st.write("**Recommended Actions:**")
                            for insight in insights:
                                st.write(f"• {insight}")
                
                # Overall business recommendations
                st.subheader("📈 Overall Business Recommendations")
                
                total_customers = len(st.session_state.rfm_clustered)
                high_value_customers = len(st.session_state.rfm_clustered[
                    st.session_state.rfm_clustered['monetary'] > st.session_state.rfm_clustered['monetary'].quantile(0.8)
                ])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Key Performance Indicators:**")
                    st.write(f"• Total Customers: {total_customers:,}")
                    st.write(f"• High-Value Customers: {high_value_customers:,} ({high_value_customers/total_customers*100:.1f}%)")
                    st.write(f"• Average Customer Value: ${st.session_state.rfm_clustered['monetary'].mean():.2f}")
                    st.write(f"• Customer Retention Rate: {len(st.session_state.rfm_clustered[st.session_state.rfm_clustered['recency'] < 90])/total_customers*100:.1f}%")
                
                with col2:
                    st.write("**Strategic Priorities:**")
                    st.write("• Focus on retaining high-value customers")
                    st.write("• Develop re-engagement campaigns for dormant customers")
                    st.write("• Create loyalty programs for regular customers")
                    st.write("• Implement personalized marketing based on segments")
                    st.write("• Monitor cluster changes over time")
            
            else:
                st.info("👆 Please perform clustering first to see business insights!")
    
    else:
        st.info("👈 Please load data using the sidebar to begin analysis!")
        
        # Show sample data format
        st.markdown("### Expected Data Format")
        sample_format = pd.DataFrame({
            'customer_id': ['CUST_0001', 'CUST_0001', 'CUST_0002'],
            'transaction_date': ['2023-01-15', '2023-02-20', '2023-01-10'],
            'monetary_value': [150.50, 89.99, 200.00]
        })
        st.dataframe(sample_format)
        
        st.markdown("""
        **Required columns:**
        - `customer_id`: Unique identifier for each customer
        - `transaction_date`: Date of transaction (YYYY-MM-DD format)
        - `monetary_value`: Transaction amount in dollars
        """)

if __name__ == "__main__":
    main()