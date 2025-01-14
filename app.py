import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.io as pio
# Set page configuration
st.set_page_config(page_title="DataFix", page_icon="🧹", layout="wide")

def plot_data_comparison(original_data, cleaned_data, plot_func):
    st.write("**Original Data:**")
    plot_func(original_data)
    st.write("**Cleaned Data:**")
    plot_func(cleaned_data)

# Apply custom CSS for styling
# def local_css(file_name):
#     try:
#         with open(file_name) as f:
#             st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
#     except FileNotFoundError:
#         st.warning("Custom CSS file not found. Default styling applied.")

# local_css("style.css")  # Ensure this file exists in your working directory

# Function to load data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            data = pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format")
        if data.empty:
            raise ValueError("The uploaded file is empty")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None
    return data

def validate_file(uploaded_file):
    """Validate the file type and extension."""
    valid_extensions = ['csv', 'xlsx']
    file_extension = uploaded_file.name.split('.')[-1]
    
    if file_extension not in valid_extensions:
        st.error(f"Invalid file format. Please upload a CSV or Excel file.")
        return False
    
    return True
# Function to clean data
def clean_data(df, missing_threshold=0.8, outlier_threshold=0.8):
    rows_before_cleaning = df.shape[0]

    # Step 1: Handle Missing Values
    # Drop columns with too many missing values
    col_missing_proportion = df.isnull().mean(axis=0)
    columns_to_retain = col_missing_proportion < missing_threshold
    df = df.loc[:, columns_to_retain]

    # Replace missing numeric values with column mean and round if the column contains integers
    numeric_cols = df.select_dtypes(include=np.number)
    if not numeric_cols.empty:
        for col in numeric_cols.columns:
            mean_value = df[col].mean()
            if pd.api.types.is_integer_dtype(df[col]):
                mean_value = round(mean_value)  # Round to nearest integer if column is integer type
            df[col].fillna(mean_value, inplace=True)

    # Replace missing categorical values with column mode
    categorical_cols = df.select_dtypes(include="object")
    if not categorical_cols.empty:
        for col in categorical_cols.columns:
            mode_value = df[col].mode()[0] if not df[col].mode().empty else ""
            df[col].fillna(mode_value, inplace=True)

    # Step 2: Handle Outliers (Flag or Cap Outliers)
    if not numeric_cols.empty:
        Q1 = numeric_cols.quantile(0.25)
        Q3 = numeric_cols.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap numeric values to bounds
        for col in numeric_cols.columns:
            df[col] = np.where(df[col] < lower_bound[col], lower_bound[col], df[col])
            df[col] = np.where(df[col] > upper_bound[col], upper_bound[col], df[col])

    rows_after_cleaning = df.shape[0]
    data_removed_percentage = ((rows_before_cleaning - rows_after_cleaning) / rows_before_cleaning) * 100

    return df, data_removed_percentage
#preview
def data_preview(df):
    # Separate numerical and categorical columns
    numeric_cols = df.select_dtypes(include='number')
    categorical_cols = df.select_dtypes(exclude='number')

    # Create a DataFrame for preview
    numeric_preview = pd.DataFrame({
        'Column Name': numeric_cols.columns,
        'Data Type': numeric_cols.dtypes,
        'Category': 'Numerical'
    })

    categorical_preview = pd.DataFrame({
        'Column Name': categorical_cols.columns,
        'Data Type': categorical_cols.dtypes,
        'Category': 'Categorical'
    })

    # Concatenate the previews
    preview = pd.concat([numeric_preview, categorical_preview], ignore_index=True)
    preview.index += 1  # To start index from 1 for better readability
    
    # Display the preview
    print("\nData Preview:")
    print(preview.to_string(index=True, header=True))

    return preview

# Visualization functions
def plot_missing_values(data):
    try:
        missing_data = data.isnull().sum().reset_index()
        missing_data.columns = ["Column", "Missing"]
        chart = (
            alt.Chart(missing_data)
            .mark_bar(color="steelblue")
            .encode(
                x=alt.X("Column", sort=alt.EncodingSortField("Missing", order="descending")),
                y="Missing",
                tooltip=["Column", "Missing"],
            )
            .properties(title="Missing Values per Column")
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating missing values chart: {e}")

def plot_outliers(data):
    try:
        numeric_data = data.select_dtypes(include=np.number)
        if not numeric_data.empty:
            melted_data = numeric_data.melt(var_name="Column", value_name="Value")
            chart = (
                alt.Chart(melted_data)
                .mark_boxplot(size=20)
                .encode(
                    x=alt.X("Column:N", title="Columns"),
                    y=alt.Y("Value:Q", title="Values"),
                    tooltip=["Column", "Value"],
                )
                .properties(title="Outliers Across Numeric Columns (Boxplot)")
            )
            st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating outliers chart: {e}")

def plot_distribution(data):
    try:
        numeric_data = data.select_dtypes(include=np.number)
        if not numeric_data.empty:
            # Create two columns for side-by-side display
            num_columns = 2
            columns = st.columns(num_columns)
            
            # Create two plots for each row of numeric columns
            for i, column in enumerate(numeric_data.columns):
                fig, ax = plt.subplots(figsize=(6, 4))  # Set a smaller size for each plot
                sns.histplot(numeric_data[column], kde=True, color='blue', stat='density', linewidth=0, ax=ax)
                ax.set_title(f"Distribution of {column} with Normal Distribution Curve")
                ax.set_xlabel(column)
                ax.set_ylabel("Density")
                
                # Ensure the plot layout is tight and does not overlap
                fig.tight_layout()
                
                # Display the plot in the correct column
                columns[i % num_columns].pyplot(fig)  # Display alternating between the two columns
        else:
            st.warning("No numeric data available for distribution plot.")
    except Exception as e:
        st.error(f"Error creating distribution plot: {e}")

def export_chart(fig):
    # Save the chart as a PNG
    pio.write_image(fig, 'plot.png')
    st.download_button("Download Plot as PNG", data=open("plot.png", "rb"), file_name="plot.png", mime="image/png")

    # Save the chart as a PDF
    pio.write_image(fig, 'plot.pdf')
    st.download_button("Download Plot as PDF", data=open("plot.pdf", "rb"), file_name="plot.pdf", mime="application/pdf")

# Streamlit app interface
# st.markdown(
#     """
#     <div style="text-align: center; padding: 20px; background-color: #eaf7ff; border-radius: 10px;">
#         <h1 style="color: #0073e6;">🧹 Data Cleaning and Visualization Tool</h1>
#         <p>Upload your dataset to clean, visualize, and compare data with simple and intuitive charts.</p>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

# # File upload
# uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

# # Initialize session state variables
# if "cleaned_data" not in st.session_state:
#     st.session_state.cleaned_data = None
# if "original_data" not in st.session_state:
#     st.session_state.original_data = None

# if uploaded_file is not None:
#     st.session_state.original_data = load_data(uploaded_file)
#     if st.session_state.original_data is not None:
#         st.success("🎉 File uploaded successfully!")
#         st.write(f"**Original Dataset:** {st.session_state.original_data.shape[0]} rows, {st.session_state.original_data.shape[1]} columns")


       # Sidebar actions
with st.sidebar:
    st.markdown("### Actions")
    if "home_action" not in st.session_state:
        st.session_state.home_action = True
    if "preview_data_action" not in st.session_state:
        st.session_state.preview_data_action = False
    if "clean_data_action" not in st.session_state:
        st.session_state.clean_data_action = False
    if "visualize_data_action" not in st.session_state:
        st.session_state.visualize_data_action = False
    if "compare_data_action" not in st.session_state:
        st.session_state.compare_data_action = False
    if "self_guide_action" not in st.session_state:
        st.session_state.self_guide_action = False  # Initialize the guide action    
    

    # Buttons that set session state variables
    if st.button("🏠 Home"):
        st.session_state.home_action = True
        st.session_state.preview_data_action = False
        st.session_state.clean_data_action = False
        st.session_state.visualize_data_action = False
        st.session_state.compare_data_action = False
        st.session_state.self_guide_action = False 

    if st.button("🔍 Data Preview"):
        st.session_state.home_action = False
        st.session_state.preview_data_action = True
        st.session_state.clean_data_action = False
        st.session_state.visualize_data_action = False
        st.session_state.compare_data_action = False
        st.session_state.self_guide_action = False  #
    if st.button("✨ Clean Data"):
        st.session_state.home_action = False
        st.session_state.clean_data_action = True
        st.session_state.preview_data_action = False
        st.session_state.visualize_data_action = False
        st.session_state.compare_data_action = False
        st.session_state.self_guide_action = False  #

    if st.button("📊 Visualize Data"):
        st.session_state.home_action = False
        st.session_state.visualize_data_action = True
        st.session_state.preview_data_action = False
        st.session_state.clean_data_action = False
        st.session_state.compare_data_action = False
        st.session_state.self_guide_action = False  #

    if st.button("🔄 Compare Original vs Cleaned Data"):
        st.session_state.home_action = False
        st.session_state.compare_data_action = True
        st.session_state.preview_data_action = False
        st.session_state.clean_data_action = False
        st.session_state.visualize_data_action = False
        st.session_state.self_guide_action = False  #

    if st.button("🤖 Self Guide"):
        st.session_state.home_action = False
        st.session_state.self_guide_action = True
        st.session_state.preview_data_action = False
        st.session_state.clean_data_action = False
        st.session_state.visualize_data_action = False
        st.session_state.compare_data_action = False
    
        # Data Preview
if st.session_state.preview_data_action:
    st.markdown("### Dataset Preview")
    st.dataframe(st.session_state.original_data, use_container_width=True)
    st.markdown("### Cleaned Data Preview")
    st.dataframe(st.session_state.cleaned_data,use_container_width=True)
   
    # Display column categories and data types for the cleaned data
    st.markdown("#### Column Categories for Data")
    cleaned_preview = data_preview(st.session_state.original_data)
    st.dataframe(cleaned_preview, use_container_width=True)

    st.session_state.data_preview = True

# Clean Data
if st.session_state.clean_data_action:
    st.session_state.cleaned_data, removed_percent = clean_data(st.session_state.original_data)
    if st.session_state.cleaned_data is not None:
        st.success(f"Data cleaned successfully! {removed_percent:.2f}% of data was removed.")
        st.download_button(
            label="Download Cleaned Data",
            data=st.session_state.cleaned_data.to_csv(index=False),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
        st.markdown("### Cleaned Data Preview")
        st.dataframe(st.session_state.cleaned_data,use_container_width=True)
        st.balloons()

# Visualize Data
if st.session_state.visualize_data_action:
    st.markdown("### Data Visualizations")
    st.write("**1. Missing Values**")
    plot_missing_values(st.session_state.original_data)

    st.write("**2. Outliers**")
    plot_outliers(st.session_state.original_data)

    st.write("**3. Normal Distribution**")
    plot_distribution(st.session_state.original_data)
    
    st.write("**4. Correlation Heatmap**")
    try:
        numeric_data = st.session_state.original_data.select_dtypes(include=np.number)
        if not numeric_data.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No numeric data available for correlation heatmap.")
    except Exception as e:
        st.error(f"Error creating heatmap: {e}")

    # 5. Scatter Plot
    st.write("**5. Scatter Plot (Select Features)**")
    try:
        numeric_columns = st.session_state.original_data.select_dtypes(include=np.number).columns
        if len(numeric_columns) >= 2:
            x_col = st.selectbox("Select X-axis Feature", numeric_columns)
            y_col = st.selectbox("Select Y-axis Feature", numeric_columns)
            scatter_chart = (
                alt.Chart(st.session_state.original_data)
                .mark_circle(size=60, opacity=0.6)
                .encode(
                    x=alt.X(x_col, title=f"{x_col} (X-axis)"),
                    y=alt.Y(y_col, title=f"{y_col} (Y-axis)"),
                    tooltip=[x_col, y_col],
                )
                .properties(title=f"Scatter Plot: {x_col} vs {y_col}")
            )
            st.altair_chart(scatter_chart, use_container_width=True)
        else:
            st.warning("Scatter plot requires at least two numeric columns.")
    except Exception as e:
        st.error(f"Error creating scatter plot: {e}")

    # 6. Bar Chart for Categorical Data
    st.write("**6. Bar Chart for Categorical Data**")
    try:
        categorical_columns = st.session_state.original_data.select_dtypes(include="object").columns
        if len(categorical_columns) > 0:
            cat_col = st.selectbox("Select a Categorical Feature", categorical_columns)
            bar_chart = (
                alt.Chart(st.session_state.original_data)
                .mark_bar(color="green", opacity=0.7)
                .encode(
                    x=alt.X(cat_col, title=f"{cat_col} Categories", sort="descending"),
                    y=alt.Y("count()", title="Count"),
                    tooltip=[cat_col, "count()"],
                )
                .properties(title=f"Bar Chart: {cat_col}")
            )
            st.altair_chart(bar_chart, use_container_width=True)
        else:
            st.warning("No categorical data available for bar chart.")
    except Exception as e:
        st.error(f"Error creating bar chart: {e}")

    # 7. Line Plot (Trends)
    st.write("**7. Line Plot for Trends**")
    try:
        if len(numeric_columns) > 1:
            line_col = st.selectbox("Select Feature for Trend", numeric_columns)
            st.line_chart(st.session_state.original_data[line_col])
        else:
            st.warning("Line plot requires at least one numeric column.")
    except Exception as e:
        st.error(f"Error creating line plot: {e}")

    # 8. Pair Plot
    st.write("**8. Pair Plot of All Numeric Features**")
    try:
        if not numeric_data.empty:
            fig = sns.pairplot(st.session_state.original_data, diag_kind="kde", markers="+")
            st.pyplot(fig)
            export_chart(fig)
        else:
            st.warning("No numeric data available for pair plot.")
    except Exception as e:
        st.error(f"Error creating pair plot: {e}")

       
    
# Compare Data
if st.session_state.compare_data_action:
    if st.session_state.cleaned_data is not None:
        comparison_option = st.selectbox(
            "Select a comparison metric:",
            ["Missing Values", "Outliers", "Rows Removed Percentage", "Columns Retained"]
        )

        if comparison_option == "Missing Values":
            st.markdown("### Missing Values Comparison")
            st.markdown("**Original Data Missing Values:**")
            plot_missing_values(st.session_state.original_data)

            st.markdown("**Cleaned Data Missing Values:**")
            plot_missing_values(st.session_state.cleaned_data)

        elif comparison_option == "Outliers":
            st.markdown("### Outliers Comparison")
            st.write("**Original Data Outliers:**")
            plot_outliers(st.session_state.original_data)

            st.write("**Cleaned Data Outliers:**")
            plot_outliers(st.session_state.cleaned_data)

        elif comparison_option == "Rows Removed Percentage":
            st.markdown("### Rows Removed Percentage")
            rows_original = st.session_state.original_data.shape[0]
            rows_cleaned = st.session_state.cleaned_data.shape[0]
            rows_removed = rows_original - rows_cleaned
            percent_removed = (rows_removed / rows_original) * 100
            st.write(f"**Original Dataset Rows:** {rows_original}")
            st.write(f"**Cleaned Dataset Rows:** {rows_cleaned}")
            st.write(f"**Rows Removed:** {rows_removed} ({percent_removed:.2f}%)")
            st.progress(percent_removed / 100)
        elif comparison_option=="Plot data Comaparison":
            plot_data_comparison(st.session_state.original_data, st.session_state.cleaned_data, plot_missing_values)


        elif comparison_option == "Columns Retained":
            st.markdown("### Columns Retained")
            original_columns = set(st.session_state.original_data.columns)
            cleaned_columns = set(st.session_state.cleaned_data.columns)
            columns_removed = original_columns - cleaned_columns
            st.write(f"**Original Dataset Columns:** {', '.join(original_columns)}")
            st.write(f"**Cleaned Dataset Columns:** {', '.join(cleaned_columns)}")
            if columns_removed:
                st.write(f"**Columns Removed:** {', '.join(columns_removed)}")
            else:
                st.success("All columns were retained during cleaning.")
    else:
        st.warning("Please clean the data first!")
if st.session_state.self_guide_action:
    st.markdown(
        """
        <div style="text-align: center; padding: 20px; background-color: #eaf7ff; border-radius: 10px;">
            <h1 style="color: #0073e6;">🤖 Self Guide: How to Clean Your Data</h1>
            <p>Follow the steps below to clean your data effectively:</p>
            <ul style="text-align: left;">
                <li><b>Step 1:</b> Upload your dataset using the "Upload" button.</li>
                <li><b>Step 2:</b> Use the "Clean Data" button to clean missing values and handle outliers.</li>
                <li><b>Step 3:</b> View the cleaned data and visualize key statistics and plots.</li>
                <li><b>Step 4:</b> Download the cleaned dataset after it's ready.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
if st.session_state.home_action:
    st.markdown(
        """
        <div style="text-align: center; padding: 20px; background-color: #eaf7ff; border-radius: 10px;">
            <h1 style="color: #0073e6;">🧹 Data Cleaning and Visualization Tool</h1>
            <p>Upload your dataset to clean, visualize, and compare data with simple and intuitive charts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

    # Initialize session state variables for data
    if "cleaned_data" not in st.session_state:
        st.session_state.cleaned_data = None
    if "original_data" not in st.session_state:
        st.session_state.original_data = None

    if uploaded_file is not None:
        st.session_state.original_data = load_data(uploaded_file)
        if st.session_state.original_data is not None:
            st.success("🎉 File uploaded successfully!")
            st.write(f"**Original Dataset:** {st.session_state.original_data.shape[0]} rows, {st.session_state.original_data.shape[1]} columns")
 # Include the following code within the Streamlit actions and flow:



