
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Employee Performance & Retention Dashboard",
                   page_icon="📊",
                   layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("employees_dataset.csv")
    df.drop_duplicates(inplace=True)
    df["education"] = df["education"].fillna(df["education"].mode()[0])
    df["previous_year_rating"] = df["previous_year_rating"].fillna(
        df["previous_year_rating"].median()
    )
    return df

df = load_data()

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select Section",
    ["Overview","Univariate","Bivariate","Multivariate",
     "Predictive Insights","Key Findings","Recommendations"]
)

# Filters
st.sidebar.header("Filters")
dept = st.sidebar.multiselect(
    "Department",
    sorted(df["department"].unique()),
    default=sorted(df["department"].unique())
)

edu = st.sidebar.multiselect(
    "Education",
    sorted(df["education"].unique()),
    default=sorted(df["education"].unique())
)

filtered_df = df[
    (df["department"].isin(dept)) &
    (df["education"].isin(edu))
]

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.title("Employee Performance & Retention Dashboard")

    c1,c2 = st.columns(2)
    c3,c4 = st.columns(2)

    c1.metric("Employees", len(filtered_df))
    c2.metric("Average Age", round(filtered_df["age"].mean(),2))
    c3.metric("Average Service Length", round(filtered_df["length_of_service"].mean(),2))
    c4.metric("Average Training Score", round(filtered_df["avg_training_score"].mean(),2))

    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(20), use_container_width=True)

# ---------------- UNIVARIATE ----------------
elif page == "Univariate":
    st.title("Univariate Analysis")

    charts = [
        px.histogram(filtered_df,x="age",title="Age Distribution"),
        px.histogram(filtered_df,x="gender",title="Gender Distribution"),
        px.histogram(filtered_df,x="education",title="Education Distribution"),
        px.histogram(filtered_df,x="department",title="Department Distribution"),
        px.histogram(filtered_df,x="region",title="Region Distribution"),
        px.histogram(filtered_df,x="recruitment_channel",title="Recruitment Channel"),
        px.histogram(filtered_df,x="no_of_trainings",title="Number of Trainings"),
        px.histogram(filtered_df,x="avg_training_score",title="Training Score Distribution"),
        px.histogram(filtered_df,x="previous_year_rating",title="Previous Year Rating"),
        px.histogram(filtered_df,x="KPIs_met_more_than_80",title="KPI Achievement"),
        px.histogram(filtered_df,x="awards_won",title="Awards Won"),
        px.histogram(filtered_df,x="length_of_service",title="Length of Service")
    ]

    for fig in charts:
        st.plotly_chart(fig, use_container_width=True)

# ---------------- BIVARIATE ----------------
elif page == "Bivariate":
    st.title("Bivariate Analysis")

    figs = [
        px.scatter(filtered_df,x="age",y="length_of_service",
                   title="Age vs Length of Service"),
        px.box(filtered_df,x="education",y="length_of_service",
               title="Education vs Length of Service"),
        px.box(filtered_df,x="department",y="avg_training_score",
               title="Department vs Training Score"),
        px.box(filtered_df,x="KPIs_met_more_than_80",y="avg_training_score",
               title="KPI Achievement vs Training Score"),
        px.box(filtered_df,x="awards_won",y="avg_training_score",
               title="Awards vs Training Score"),
        px.box(filtered_df,x="previous_year_rating",y="avg_training_score",
               title="Previous Rating vs Training Score"),
        px.scatter(filtered_df,x="length_of_service",y="avg_training_score",
                   title="Length of Service vs Training Score"),
        px.box(filtered_df,x="department",y="KPIs_met_more_than_80",
               title="Department vs KPI Achievement")
    ]

    for fig in figs:
        st.plotly_chart(fig, use_container_width=True)

# ---------------- MULTIVARIATE ----------------
elif page == "Multivariate":
    st.title("Multivariate Analysis")

    fig1 = px.scatter(
        filtered_df,
        x="age",
        y="avg_training_score",
        color="previous_year_rating",
        title="Age + Training Score + Previous Rating"
    )

    fig2 = px.box(
        filtered_df,
        x="awards_won",
        y="avg_training_score",
        color="KPIs_met_more_than_80",
        title="Awards + KPI + Training Score"
    )

    dept_avg = filtered_df.groupby(
        ["department","KPIs_met_more_than_80"]
    )["avg_training_score"].mean().reset_index()

    fig3 = px.bar(
        dept_avg,
        x="department",
        y="avg_training_score",
        color="KPIs_met_more_than_80",
        title="Department + KPI + Training Score"
    )

    st.plotly_chart(fig1,use_container_width=True)
    st.plotly_chart(fig2,use_container_width=True)
    st.plotly_chart(fig3,use_container_width=True)

    corr = filtered_df.corr(numeric_only=True)

    heatmap = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu"
    )

    st.subheader("Correlation Heatmap")
    st.plotly_chart(heatmap,use_container_width=True)

# ---------------- PREDICTIVE ----------------
elif page == "Predictive Insights":
    st.title("Predictive Insights")

    high_perf = filtered_df[
        (filtered_df["KPIs_met_more_than_80"] == 1) &
        (filtered_df["previous_year_rating"] >= 4)
    ]

    risk = filtered_df[
        (filtered_df["KPIs_met_more_than_80"] == 0) &
        (filtered_df["awards_won"] == 0)
    ]

    c1,c2 = st.columns(2)

    c1.metric("High Performers", len(high_perf))
    c2.metric("Retention Risk", len(risk))

    conditions = [
        (filtered_df['KPIs_met_more_than_80']==1) &
        (filtered_df['awards_won']==1),
        (filtered_df['KPIs_met_more_than_80']==1),
        (filtered_df['KPIs_met_more_than_80']==0)
    ]

    labels = [
        "Star Employee",
        "Good Performer",
        "Needs Improvement"
    ]

    filtered_df["segment"] = np.select(
        conditions,
        labels,
        default="Average Performer"
    )

    seg = filtered_df["segment"].value_counts().reset_index()
    seg.columns=["Segment","Count"]

    st.plotly_chart(
        px.bar(seg,x="Segment",y="Count",
               title="Employee Segmentation"),
        use_container_width=True
    )

# ---------------- FINDINGS ----------------
elif page == "Key Findings":
    st.title("Key Findings")

    findings = [
        "Higher training scores are associated with better KPI achievement.",
        "Award-winning employees generally demonstrate stronger performance.",
        "Employees with higher education tend to remain longer.",
        "Training positively influences employee performance.",
        "Department-level differences affect performance and retention.",
        "High performers show stronger ratings and KPI achievement."
    ]

    for f in findings:
        st.success(f)

# ---------------- RECOMMENDATIONS ----------------
elif page == "Recommendations":
    st.title("HR Recommendations")

    recs = [
        "Strengthen department-specific training programs.",
        "Increase employee recognition initiatives.",
        "Provide coaching for low KPI performers.",
        "Create leadership pathways for high performers.",
        "Improve career growth opportunities.",
        "Implement retention strategies for at-risk employees.",
        "Monitor training effectiveness regularly."
    ]

    for r in recs:
        st.info(r)

    st.subheader("Conclusion")
    st.write(
        "Training scores, KPI achievement, awards, and ratings significantly "
        "influence employee performance. Improving training, recognition, "
        "and career development can enhance retention and organizational success."
    )
