import streamlit as st
import pandas as pd
import joblib
import numpy as np
import altair as alt

@st.cache(allow_output_mutation=True)
def load_model():
# Load the trained model
	return joblib.load('enbloc_model_0204.pkl')

@st.cache(allow_output_mutation=True)
def load_data():
# Load the dataset
	return pd.read_csv('Cleaned_Final_Comb_Dataset_1830.csv')


def likelihood(score):
    if score < 0.2:
        return "Very Unlikely"
    elif score < 0.4:
        return "Unlikely"
    elif score < 0.6:
        return "Possible"
    elif score < 0.8:
        return "Likely"
    else:
        return "Very Likely"


# Calculate en bloc probabilities for all projects
def calculate_probabilities(df, model):
	df['Enbloc Probability'] = df.apply(lambda x: model.predict_proba(x[['Land Size', 'Master Plan GFA', 'Plot Ratio', 'Property Type', 'Tenure', 'District', 'Age', 'Number of Units', 'Average Price', 'Historical High', 'Historical Low', 'Distance to MRT']].values.reshape(1, -1))[0][1], axis=1)
	return df

# Define a function to get project details
def get_project_details(project_name, df):
	project = df[df['Project Name'] == project_name].iloc[0]
	return project

st.title('En Bloc Prediction App')

model = load_model()
df = load_data()

df = calculate_probabilities(df, model)

# Visualizations
st.header('Visualizations')

df['En Bloc Likelihood'] = df['Enbloc Probability'].apply(likelihood)
likelihood_count = df.groupby(['District', 'En Bloc Likelihood']).size().reset_index(name='Count')
chart_likelihood = alt.Chart(likelihood_count).mark_bar().encode(
	x=alt.X('District:N', title='District'),
	y=alt.Y('Count:Q', title='Number of Condos'),
	color='En Bloc Likelihood'
)

st.altair_chart(chart_likelihood, use_container_width=True)

# Top en bloc candidates
st.header('Top En Bloc Candidates')
planning_areas = df['Planning Area'].unique()
selected_area = st.selectbox('Select a planning area:', planning_areas)
top_n = 10
top_candidates = df[df['Planning Area'] == selected_area].nlargest(top_n, 'Enbloc Probability')[['Project Name']]
top_candidates.reset_index(drop=True, inplace=True)
top_candidates.index = top_candidates.index + 1
st.write(top_candidates)

# Comparison of multiple condos
st.header('Compare Multiple Condos')
project_names = df['Project Name'].unique()
selected_projects = st.multiselect('Select condos to compare:', project_names)
comparison_df = df[df['Project Name'].isin(selected_projects)][['Project Name', 'Tenure', 'District', 'Enbloc Probability']]

# Convert Tenure to Freehold or Leasehold
comparison_df['Tenure'] = comparison_df['Tenure'].apply(lambda x: 'Freehold' if x == 1 else 'Leasehold')

# Convert Enbloc Probability to likelihood score
comparison_df['En Bloc Likelihood'] = comparison_df['Enbloc Probability'].apply(likelihood)
comparison_df['En Bloc Probability'] = comparison_df['Enbloc Probability'].apply(lambda x: f"{x*100:.2f}%")
st.write(comparison_df.set_index('Project Name'))

# Custom input (for use with data from https://www.edgeprop.sg/)
st.header('Custom Input')
st.markdown("(Based on URA sales data in the last 12 months. Otherwise, based on latest transaction. May not be representative.)")
st.markdown("(To be used with information from edgeprop.sg , distance to nearest MRT can be found using Google Map.)")

custom_input = {
'Land Size': st.number_input('Land Size (sqm):', value=1000),
'Master Plan GFA': st.number_input('Master Plan GFA (sqm):', value=5000),
'Plot Ratio': st.number_input('Plot Ratio:', value=1.5),
'Property Type': st.selectbox('Property Type:', options=[(0, 'Apartment'), (1, 'Condominium')], format_func=lambda x: x[1])[0],
'Tenure': st.selectbox('Tenure:', options=[(0, 'Leasehold'), (1, 'Freehold')], format_func=lambda x: x[1])[0],
'District': st.number_input('District:', min_value=1, max_value=28, value=1),
'Age': st.number_input('Age (estimated, in 2043):', value=10),
'Number of Units': st.number_input('Number of Units:', value=100),
'Average Price': st.number_input('Average Price* (psf):', value=1000),
'Historical High': st.number_input('Historical High (psf):', value=1500),
'Historical Low': st.number_input('Historical Low (psf):', value=650),
'Distance to MRT': st.number_input('Distance to MRT (KM):', value=0.01)
}

custom_df = pd.DataFrame(custom_input, index=[0])

# Make the prediction
custom_en_bloc_prediction = model.predict_proba(custom_df)[0][1]

st.subheader("En Bloc Probability for Custom Input")
st.write(f"{custom_en_bloc_prediction*100:.2f}%")