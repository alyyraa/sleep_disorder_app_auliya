import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

def display_dataset_overview(df):
    """
    Display basic information about the dataset
    """
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Duplicates", df.duplicated().sum())
    
    # Display first few rows
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10))
    
    # Display data types and basic statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values
        })
        st.dataframe(dtype_df)
    
    with col2:
        st.subheader("📊 Summary Statistics")
        st.dataframe(df.describe())

def plot_correlation_heatmap(df):
    """
    Create correlation heatmap for numerical features
    """
    st.subheader("🔥 Correlation Heatmap")
    
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        correlation_matrix = df[numerical_cols].corr()
        
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   ax=ax)
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Not enough numerical columns for correlation analysis.")

def plot_distribution_plots(df):
    """
    Create distribution plots for key numerical features
    """
    st.subheader("📈 Distribution Plots")
    
    # Key numerical features to plot
    key_features = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
                   'Stress Level', 'Heart Rate', 'Daily Steps']
    
    # Filter existing columns
    available_features = [col for col in key_features if col in df.columns]
    
    if available_features:
        # Create subplots
        n_cols = 2
        n_rows = (len(available_features) + 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature in enumerate(available_features):
            if i < len(axes):
                sns.histplot(data=df, x=feature, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {feature}')
                axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(available_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No suitable numerical features found for distribution plots.")

def plot_sleep_disorder_analysis(df):
    """
    Analyze sleep disorder distribution and relationships
    """
    st.subheader("😴 Sleep Disorder Analysis")
    
    if 'Sleep Disorder' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sleep disorder distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            sleep_counts = df['Sleep Disorder'].value_counts()
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            wedges, texts, autotexts = ax.pie(sleep_counts.values, 
                                            labels=sleep_counts.index,
                                            autopct='%1.1f%%',
                                            colors=colors,
                                            startangle=90)
            
            ax.set_title('Sleep Disorder Distribution')
            st.pyplot(fig)
        
        with col2:
            # Sleep disorder by gender (if available)
            if 'Gender' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                pd.crosstab(df['Gender'], df['Sleep Disorder']).plot(kind='bar', ax=ax)
                ax.set_title('Sleep Disorder by Gender')
                ax.set_xlabel('Gender')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
    else:
        st.warning("Sleep Disorder column not found in the dataset.")

def plot_stress_level_analysis(df):
    """
    Analyze stress level patterns
    """
    st.subheader("😰 Stress Level Analysis")
    
    if 'Stress Level' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Stress level distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(data=df, x='Stress Level', bins=10, kde=True, ax=ax)
            ax.set_title('Stress Level Distribution')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            # Stress level vs Sleep Quality
            if 'Quality of Sleep' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(data=df, x='Quality of Sleep', y='Stress Level', 
                              hue='Sleep Disorder' if 'Sleep Disorder' in df.columns else None,
                              ax=ax)
                ax.set_title('Stress Level vs Sleep Quality')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
    else:
        st.warning("Stress Level column not found in the dataset.")

def plot_interactive_scatter(df):
    """
    Create interactive scatter plots using Plotly
    """
    st.subheader("🎯 Interactive Analysis")
    
    # Select features for scatter plot
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox("Select X-axis", numerical_cols, index=0)
        with col2:
            y_axis = st.selectbox("Select Y-axis", numerical_cols, index=1 if len(numerical_cols) > 1 else 0)
        
        # Create interactive scatter plot
        if 'Sleep Disorder' in df.columns:
            fig = px.scatter(df, x=x_axis, y=y_axis, 
                           color='Sleep Disorder',
                           title=f'{x_axis} vs {y_axis}',
                           hover_data=['Age', 'Gender'] if 'Age' in df.columns and 'Gender' in df.columns else None)
        else:
            fig = px.scatter(df, x=x_axis, y=y_axis,
                           title=f'{x_axis} vs {y_axis}')
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough numerical columns for scatter plot analysis.")

def plot_feature_importance_by_target(df):
    """
    Show feature relationships with target variables
    """
    st.subheader("🎯 Feature Relationships")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'Sleep Disorder' in df.columns and len(numerical_cols) > 0:
        # Box plots for numerical features vs Sleep Disorder
        n_cols = 2
        n_features = min(6, len(numerical_cols))  # Limit to 6 features
        n_rows = (n_features + 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature in enumerate(numerical_cols[:n_features]):
            if i < len(axes):
                sns.boxplot(data=df, x='Sleep Disorder', y=feature, ax=axes[i])
                axes[i].set_title(f'{feature} by Sleep Disorder')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)

def generate_insights(df):
    """
    Generate automated insights from the data
    """
    st.subheader("💡 Key Insights")
    
    insights = []
    
    # Basic dataset insights
    insights.append(f"📊 Dataset contains {len(df)} records with {len(df.columns)} features")
    
    # Sleep disorder insights
    if 'Sleep Disorder' in df.columns:
        disorder_counts = df['Sleep Disorder'].value_counts()
        most_common = disorder_counts.index[0]
        percentage = (disorder_counts.iloc[0] / len(df)) * 100
        insights.append(f"😴 Most common condition: {most_common} ({percentage:.1f}% of cases)")
    
    # Age insights
    if 'Age' in df.columns:
        avg_age = df['Age'].mean()
        insights.append(f"👥 Average age of participants: {avg_age:.1f} years")
    
    # Sleep quality insights
    if 'Quality of Sleep' in df.columns:
        avg_quality = df['Quality of Sleep'].mean()
        insights.append(f"😴 Average sleep quality score: {avg_quality:.1f}/10")
    
    # Stress level insights
    if 'Stress Level' in df.columns:
        avg_stress = df['Stress Level'].mean()
        high_stress_pct = (df['Stress Level'] >= 7).mean() * 100
        insights.append(f"😰 Average stress level: {avg_stress:.1f}/10")
        insights.append(f"🚨 {high_stress_pct:.1f}% of participants have high stress (≥7)")
    
    # Display insights
    for insight in insights:
        st.write(f"• {insight}")

def create_comprehensive_eda(df):
    """
    Create a comprehensive EDA report
    """
    st.title("🔍 Exploratory Data Analysis")
    
    # Dataset overview
    display_dataset_overview(df)
    
    st.divider()
    
    # Correlation analysis
    plot_correlation_heatmap(df)
    
    st.divider()
    
    # Distribution plots
    plot_distribution_plots(df)
    
    st.divider()
    
    # Sleep disorder analysis
    plot_sleep_disorder_analysis(df)
    
    st.divider()
    
    # Stress level analysis
    plot_stress_level_analysis(df)
    
    st.divider()
    
    # Interactive analysis
    plot_interactive_scatter(df)
    
    st.divider()
    
    # Feature relationships
    plot_feature_importance_by_target(df)
    
    st.divider()
    
    # Generate insights
    generate_insights(df)