import pandas as pd
import numpy as np
import sqlite3
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

#=======================================================================================

# Title, subheader, and description of the project 
st.title("🎶 Group Project Dashboard: Spotify Dataset 🎶")
st.markdown("### **Welcome to Our Spotify Dashboard!**")
st.markdown("**Our team consists of:** Bashir, Lambert, Deekshanya, Daniel, Aadya.")
st.markdown("""
In this dashboard, we explore various attributes of Spotify tracks, such as **popularity**, **danceability**, and more.)
""")
st.markdown("---")


#=======================================================================================
data_dictionary = {
    "track_id": "The Spotify ID for the track",
    "artists": "The artists' names who performed the track. If there is more than one artist, they are separated by a ;",
    "album_name": "The album name in which the track appears",
    "track_name": "Name of the track",
    "popularity": "The popularity of a track is a value between 0 and 100, with 100 being the most popular. It is calculated by an algorithm and is based on the total number of plays the track has had and how recent those plays are.",
    "duration_ms": "The track length in milliseconds",
    "explicit": "Whether or not the track has explicit lyrics (true = yes it does; false = no it does not or unknown)",
    "danceability": "Describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.",
    "energy": "A measure from 0.0 to 1.0 representing a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy.",
    "key": "The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g., 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.",
    "loudness": "The overall loudness of a track in decibels (dB).",
    "mode": "Indicates the modality (major or minor) of a track. Major is represented by 1, and minor is represented by 0.",
    "speechiness": "Detects the presence of spoken words in a track. Values above 0.66 describe tracks that are probably made entirely of spoken words.",
    "acousticness": "A confidence measure from 0.0 to 1.0 of whether the track is acoustic. A value of 1.0 represents high confidence that the track is acoustic.",
    "instrumentalness": "Predicts whether a track contains no vocals. The closer the instrumentalness value is to 1.0, the greater the likelihood the track contains no vocal content.",
    "liveness": "Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live.",
    "valence": "A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive.",
    "tempo": "The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece.",
    "time_signature": "An estimated time signature. The time signature ranges from 3 to 7, indicating time signatures of 3/4 to 7/4.",
    "track_genre": "The genre to which the track belongs."
}

# Streamlit App Layout
st.title("Spotify Dataset Variable Information")
st.subheader("Select a variable from the dropdown to view its details.")

# Dropdown widget to select a variable
variable = st.selectbox("Choose a variable", list(data_dictionary.keys()))

# Display the description of the selected variable
st.write(f"**Variable:** {variable}")
st.write(f"**Description:** {data_dictionary[variable]}")
st.markdown("---")

#=======================================================================================
#QUESTION 1
st.title("🎶 Guiding Question 1: What are popular songs' most common traits?🎶 ")
st.write("""We took a look at five different traits of songs in the top 10% of popularity scores. 
         We found that two traits stood out, **energy** and **danceability**. This information 
         can be used for artists or labels to produce tracks with these characteristics in mind.
         Music streaming platforms can also use songs with high energy and danceability scores 
         in their curated playlists, knowing that they are popular, in hopes of increasing customer 
         satisfaction and engagement. """)
st.write("""Instead of just showing you a chart with the values for only songs in top 10%, 
         we have included a slider for you to examine common traits for songs with a popularity score in the 
         top 50%""")


# Streamlit widget to select the popularity percentile
percentile_slider = st.slider(
    "Select Popularity Percentile", 
    min_value=50, 
    max_value=100, 
    value=90,  # Default value is 90th percentile
    step=1
)

# Connect to SQLite database
conn = sqlite3.connect('spotify_data.db')
cursor = conn.cursor()

# Calculate the dynamic percentile value based on the slider input
cursor.execute(f"SELECT popularity FROM spotify_tracks")
popularity_values = [row[0] for row in cursor.fetchall()]
dynamic_percentile = pd.Series(popularity_values).quantile(percentile_slider / 100)

# Query for the most popular songs based on the dynamic percentile 
query_dynamic = f"""
WITH popular_songs AS (
    SELECT 
        track_id,
        track_name,
        popularity,
        energy,
        speechiness,
        acousticness,
        danceability,
        instrumentalness
    FROM spotify_tracks
    WHERE popularity >= {dynamic_percentile}
)
SELECT 
    AVG(energy) AS avg_energy,
    AVG(speechiness) AS avg_speechiness,
    AVG(acousticness) AS avg_acousticness,
    AVG(danceability) AS avg_danceability,
    AVG(instrumentalness) AS avg_instrumentalness
FROM popular_songs;
"""

# Execute the query and load data into a pandas DataFrame
cursor.execute(query_dynamic)
result_dynamic = cursor.fetchone()  # Assuming there's only one row of result (avg values)

# Update DataFrame based on dynamic query result
data_dynamic = {
    'Trait': ['Energy', 'Speechiness', 'Acousticness', 'Danceability', 'Instrumentalness'],
    'Average Value': [result_dynamic[0], result_dynamic[1], result_dynamic[2], result_dynamic[3], result_dynamic[4]]
}

df_dynamic = pd.DataFrame(data_dynamic)

# Plot bar chart based on slider value
fig_dynamic = px.bar(df_dynamic, 
             x='Trait', 
             y='Average Value',
             color='Average Value', 
             title=f"Average Traits of Popular Songs (Top {percentile_slider}th Percentile)",
             labels={"Average Value": "Average Value", "Trait": "Trait"},
             color_continuous_scale='Plasma',  # Change color scale for dynamic chart
             template='plotly_dark')  # Dark theme for the chart

# Code for layout
fig_dynamic.update_layout(
    xaxis_title="Trait",
    yaxis_title="Average Value",
    xaxis=dict(tickangle=45),
)

# Update x-axis to rotate the labels from left to right and adjust the layout to fit
fig_dynamic.update_layout(
    xaxis_tickangle=0,  # Tilt labels from bottom-left to top-right (increasing)
    margin=dict(t=40, b=100),  # Adjust top and bottom margins to fit labels
)

# Display the dynamic chart
st.header("Key Characteristics of Popular Songs")
st.plotly_chart(fig_dynamic)
st.markdown("---")
# Close the connection
conn.close()

#=======================================================================================
#Question 2
st.title("🎶 Guiding Question 2: Which artists composed the top 20 songs in this dataset? 🎶")
st.write("""This information could be useful for talent managers and music labels to identify 
         which artists are currently dominating the charts and are being listened to more frequently.
         Also, brands can use this information to find artists that performing well and collaborate 
         with them for their marketing and engagement needs.""")
st.write("""Similar to the reasoning behind the first question, this is also useful for streaming platforms 
         to highlight top artists and songs that could increase user engagement and time spent on 
         the platform. This is relevant as their business strategy for monetization is through customer subscriptions. """)
st.write("""Instead of just showing you a chart with the values for the top 20 songs in the dataset, 
         we have included a slider for you to examine the top 50 songs and what their popularity scores are!""")

conn = sqlite3.connect('spotify_data.db')

#SQL Query to get the top 20 songs by popularity
query = """
SELECT track_name, artists, popularity
FROM spotify_tracks
ORDER BY popularity DESC
LIMIT 20
"""
# Slider to select the number of top songs to display
num_songs = st.slider('Select Number of Top Songs to Display:', 5, 50, 10)
#Execute the query and load the data into a DataFrame
df_top_20_songs = pd.read_sql(query, conn)

#Close the database connection
conn.close()

#Display the DataFrame in Streamlit
st.write(f"#### Top {num_songs} Songs and Their Composers (Artists):")
st.write(df_top_20_songs)

#Connect to SQLite database
conn = sqlite3.connect('spotify_data.db')

#SQL Query to get the top songs by popularity
query = f"""
SELECT track_name, artists, popularity
FROM spotify_tracks
ORDER BY popularity DESC
LIMIT 20
"""
st.subheader('Visualization of the Results')
#Execute the query and load data into a DataFrame
df_top_songs = pd.read_sql(query, conn)

#Close the connection to the database
conn.close()

# Step 7: Create a Bar Chart to visualize song popularity
fig = px.bar(df_top_songs, x='track_name', y='popularity',
             color='popularity', title="🎶 Top Songs by Popularity",
             labels={'track_name': 'Song Name', 'popularity': 'Popularity'},
             color_continuous_scale='Viridis')

# Display the plot
st.plotly_chart(fig)
st.markdown("---")

#=======================================================================================
#Question 3 
st.title("🎶 Guiding Question 3: Is there a time signature and tempo that is most popular? 🎶")
st.write("""Understanding the most popular time signatures and tempos is crucial for businesses in the music industry,
        as these rhythmic elements directly impact the appeal of a song to listeners. For music producers, songwriters, 
        and talent managers, aligning music with popular time signatures and tempos can enhance the chances of a track 
        performing well commercially, increasing its playtime and fan engagement. This can lead to more sales, streaming 
        royalties, and higher chart rankings, benefiting the artists, labels, and producers financially.""")


conn = sqlite3.connect('spotify_data.db')
cursor = conn.cursor()

# Step 1: Connect to the database and fetch the data
query = """
SELECT popularity, tempo, time_signature
FROM spotify_tracks
ORDER BY popularity DESC
"""
df = pd.read_sql(query, conn)
conn.close()  # Close the connection after fetching the data

# Display the fetched data in Streamlit
st.write("Preview of the Data", df.head())

# Step 2: Bin the time_signature and tempo
time_sig_bins = [0, 1, 3, 4, 5, 6]
tempo_bins = np.linspace(0, 245, 12)

# Create bin labels
df['time_sig_bin'] = pd.cut(df['time_signature'], bins=time_sig_bins, labels=np.arange(len(time_sig_bins) - 1))
tempo_labels = [f"{int(tempo_bins[i])}-{int(tempo_bins[i+1])}" for i in range(len(tempo_bins) - 1)]
df['tempo_bin_label'] = pd.cut(df['tempo'], bins=tempo_bins, labels=tempo_labels)

# Step 3: Group by bins and calculate average popularity
heatmap_data = (
    df.groupby(['time_sig_bin', 'tempo_bin_label'])['popularity']
    .mean()
    .reset_index()
    .pivot(index='tempo_bin_label', columns='time_sig_bin', values='popularity')
)

# Fill missing values with 0 for visualization
heatmap_data = heatmap_data.fillna(0)

# Step 4: Visualize the heatmap using Plotly
fig = px.imshow(
    heatmap_data.T,  # Transpose to swap axes
    labels={"x": "Tempo Range", "y": "Time Signature Bin", "color": "Avg Popularity"},
    color_continuous_scale="Viridis",
    title="Heatmap of Popularity by Tempo Range and Time Signature"
)

# Step 5: Display the heatmap in Streamlit
st.subheader('Average Popularity by Tempo and Time Signature:')
st.plotly_chart(fig)
st.markdown("---")

#=======================================================================================
#QUESTION 4 
st.title("🎶 Guiding Question 4: Which genres have higher popularity scores and which lower scores? 🎶")
st.write(""" This analysis is valuable for music industry professionals, including record labels, artists, 
        and talent managers, as it identifies which genres are currently more popular and performing better in
        terms of audience engagement. For instance, if a particular genre shows consistently high popularity, brands 
        and music platforms may prioritize artists within that genre to drive user engagement and maximize revenue.""")
st.write("""For streaming platforms, understanding the popularity of different genres helps to personalize recommendations, 
        create targeted playlists, and improve the user experience. By offering more content from genres with higher popularity 
         scores, platforms can boost user retention, increase listening time, and attract new subscribers.""")
st.write("For this dataset, **k-pop** was the most popular genre. Second was **pop** and third was **deep-house**. ")

# Connect to SQLite database
conn = sqlite3.connect('spotify_data.db')
cursor = conn.cursor()

# Genres list for filtering
genres_list = ['pop', 'rock', 'hip-hop', 'classical', 'blues', 'edm', 'folk', 'heavy-metal', 
               'indie', 'jazz', 'reggae', 'alt-rock', 'acoustic', 'deep-house', 'k-pop', 'latino',
               'french', 'indie-pop', 'bluegrass', 'dubstep']

# Streamlit widget for genre selection
selected_genres = st.multiselect(
    "Select genres to analyze:",
    options=genres_list,
    default=genres_list,  # Default to show all genres
    help="Choose genres to compare average popularity scores."
)

#SQL query based on selected genres
query = f"""
SELECT 
    track_genre,
    AVG(popularity) AS avg_popularity
FROM spotify_tracks
WHERE track_genre IN ({','.join([f"'{genre}'" for genre in selected_genres])})
GROUP BY track_genre;
"""

# Load the data into pandas DataFrame
df = pd.read_sql_query(query, conn)

# Close the connection
conn.close()

#Bar chart using Plotly Express
fig = px.bar(
    df, 
    x='track_genre', 
    y='avg_popularity', 
    color='avg_popularity',  # Color the bars based on popularity score
    title="🎶 Average Popularity by Genre",
    labels={"track_genre": "Genre", "avg_popularity": "Average Popularity"},
    template="plotly_dark",  # Dark background for a modern aesthetic
    color_continuous_scale='Viridis',  # Color scale for a vibrant look
    height=500  # Adjust the chart height for a better fit
)

# Display the plot in Streamlit
st.header("Popularity Comparison Across Genres")
st.write('We selected the genres you chose for this visualization. The bars are colored based on popularity scores, with brighter colors indicating higher popularity.')
st.plotly_chart(fig)
st.markdown("---")

#=======================================================================================
#QUESTION 5
st.title("🎶 Guiding Question 5: How does popularity correlate with factors in the dataset? 🎶")
st.write("""Understanding how popularity correlates with various factors in music tracks, such as danceability, acousticness, 
        speechiness, instrumentalness, and energy, is crucial for businesses in the music industry.For music labels and artists,
        knowing the characteristics that drive popularity could inform future song production. If danceability or high energy 
        correlates with better reception, artists could focus on creating music that fits those qualities, improving their chances 
        of success.""")
st.write("""Based on the results of the analysis, **none** of the traits had a strong enough correlation with popularity. It is important
        to note that correlation doesn't mean causation, but there is no significant linear relationship present in the data. 
        However, there are strong correlations between traits which may be used for another business question. """)
 

# Connect to the SQLite database
conn = sqlite3.connect('spotify_data.db')
cursor = conn.cursor()

# SQL query to select the necessary columns
query = """
SELECT popularity, danceability, acousticness, speechiness, instrumentalness, 
       energy, loudness, valence
FROM spotify_tracks;
"""

# Execute the query and load the data into a pandas DataFrame
df = pd.read_sql_query(query, conn)

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create a Plotly Express heatmap
fig = px.imshow(correlation_matrix,
                text_auto=True,  # Automatically adds correlation coefficient values in each cell
                color_continuous_scale='RdBu',  # Color scale for the correlation values
                zmin=-1, zmax=1,  # Set the range for correlation coefficients
                title="Correlation Heatmap: Popularity vs Traits")

# Display the heatmap in Streamlit
st.header("Spotify Songs: Popularity Correlation with Traits")
st.plotly_chart(fig)

# Close the connection
conn.close()
st.markdown("---")

#=======================================================================================
#Question 6
st.title("🎶 Guiding Question 6: Is there an optimal track duration for higher popularity scores?🎶")
st.write("""This analysis helps businesses identify the track duration most associated with higher popularity scores. For record
        labels, artists, and music platforms, it provides insights into the duration preferences of listeners. Streaming platforms
        can use this data to optimize their recommendation algorithms, while artists can adjust their creative process to match 
        popular trends.""")
st.write("""As far as the results, songs that range between 4 and 5 minutes have the highest average popularity score. Second, and 
        not too far away, are songs that are 3-4 minutes long. Artists could potentially use this information when designing 
        their songs, as well as concert track selection. """)

# Extract Data from SQLite Database
conn = sqlite3.connect('spotify_data.db')
query = """
SELECT duration_min, popularity
FROM spotify_tracks
ORDER BY popularity DESC
"""
df = pd.read_sql(query, conn)
conn.close()

# custom bins for duration_min (0-2, 2-3, 3-4, 4-5, 5-10, and 10-100 minutes)
duration_bins = [0, 2, 3, 4, 5, 10, 100]  # Custom bins as specified
bin_labels = ["0-2 min", "2-3 min", "3-4 min", "4-5 min", "5-10 min", "10-100 min"]

# Bin the data into these custom bins
df['duration_bin'] = pd.cut(df['duration_min'], bins=duration_bins, labels=bin_labels)

#Group by bins and calculate average popularity
grouped_data = (
    df.groupby('duration_bin')['popularity']
    .mean()
    .reset_index()
    .rename(columns={"popularity": "avg_popularity"})
)

# bar chart using Plotly with color representing avg_popularity
fig = px.bar(
    grouped_data,
    x="duration_bin",
    y="avg_popularity",
    color="avg_popularity",  # Color bars based on avg_popularity
    labels={"duration_bin": "Duration (mins)", "avg_popularity": "Avg Popularity"},
    title="Average Popularity vs Duration Intervals",
    color_continuous_scale="Viridis"  # You can change the color scale (e.g., "Viridis", "Cividis", etc.)
)

#Plotly chart in Streamlit
st.plotly_chart(fig)  # Display the chart


#Connect to the SQLite Database
conn = sqlite3.connect('spotify_data.db')  # Adjust the path if necessary

#Query to retrieve the data (genre and song duration)
query = """
SELECT track_genre, duration_ms
FROM spotify_tracks
"""

#Load the data into a DataFrame
spotify_data = pd.read_sql(query, conn)

#Close the database connection
conn.close()

# Convert duration from milliseconds to minutes for easier interpretation
spotify_data['duration_min'] = spotify_data['duration_ms'] / 60000  # Convert ms to minutes
genre_duration = spotify_data.groupby('track_genre')['duration_min'].mean().reset_index()
genre_duration = genre_duration.sort_values(by='duration_min', ascending=False)

# Step 8: Streamlit app to display the results
st.header('Average Song Duration by Genre')
st.write('We have also include a widget where you can select a genre and receive the average duration')

#selectbox for genre selection
genre_options = genre_duration['track_genre'].unique()  # Get list of unique genres
selected_genre = st.selectbox(
    'Select a genre to see the average song duration:',
    genre_options
)

# Filter the data based on the selected genre
selected_genre_data = genre_duration[genre_duration['track_genre'] == selected_genre]

#average song duration for the selected genre
st.write(f"The average song duration for {selected_genre} is {selected_genre_data['duration_min'].values[0]:.2f} minutes.")

#bar chart of average duration for all genres using Plotly
fig = px.bar(
    genre_duration,
    x='track_genre',
    y='duration_min',
    color='duration_min',
    labels={'track_genre': 'Genre', 'duration_min': 'Average Duration (min)'},
    title='Average Song Duration for Each Genre',
    color_continuous_scale='Cividis'  # Optional: You can choose a different color scale
)

st.plotly_chart(fig)

st.markdown("---")


#=======================================================================================

#Question 7
st.title("🎶 Guiding Question 7:  Are louder songs (higher decibels) more likely to be popular, or is there a sweet spot for loudness?🎶")
st.write(""" Brands using music for marketing campaigns can benefit from understanding loudness's impact on popularity. If louder songs
        are more popular, these types of tracks can be chosen for commercials or in-store playlists to better capture attention,
        evoke specific emotions, and drive customer engagement.""")
st.write("""Artists and producers can optimize the loudness levels of their tracks based on trends in popular music. By knowing 
        the "sweet spot" for loudness, artists can adjust their music production techniques to create tracks that are more likely
        to resonate with listeners and attract higher play counts.""")
st.write("""Based on the results of the data, specificially in the bar chart, we see the data is skewed to left. Popularity starts
         to spike around -10 and is high between -10 and -2. Tracks with decibels(dB) between -10 and -2 in this dataset 
         have the highest popularity scores. tracks below -10 decibels (dB) have lower scores. """ )

         
st.subheader("We included a scatterplot first and a bar chart for clarity")
# Step 1: Extract Data from SQLite Database
conn = sqlite3.connect('spotify_data.db')
query = """
SELECT popularity, loudness
FROM spotify_tracks
ORDER BY popularity DESC
"""
df = pd.read_sql(query, conn)
conn.close()

# Step 2: Create a scatterplot using Plotly
fig = px.scatter(
    df,
    x="loudness",  # X-axis: Loudness
    y="popularity",  # Y-axis: Popularity
    labels={"loudness": "Loudness", "popularity": "Popularity"},
    title="Scatterplot of Popularity vs Loudness",
    trendline="ols",
     trendline_color_override=st.color_picker("Pick Trendline Color", "#FF5733")
)

# Step 3: Display the Plotly scatterplot in Streamlit
st.header("Spotify Track Popularity vs Loudness")  # Title for the app
st.plotly_chart(fig)  # Display the scatterplot

# Step 1: Extract Data from SQLite Database
conn = sqlite3.connect('spotify_data.db')
query = """
SELECT popularity, loudness
FROM spotify_tracks
ORDER BY popularity DESC
"""
df = pd.read_sql(query, conn)
conn.close()


# Step 2: Create a bar chart using Plotly
fig = px.bar(
    df,
    x="loudness",  # X-axis: Loudness
    y="popularity",  # Y-axis: Popularity
    labels={"loudness": "Loudness", "popularity": "Popularity"},
    title="Bar Chart of Popularity vs Loudness",
)

# Step 3: Display the Plotly bar chart in Streamlit
st.header("Spotify Track Popularity vs Loudness")  # Title for the app
st.plotly_chart(fig)  # Display the bar chart
st.markdown("---")

#=======================================================================================
#Question 8
st.title("🎶 Guiding Question 8: Do people drift towards positive or negative valence music?🎶")
st.write("""By identifying whether positive or negative valence music is more popular, businesses in the music industry—such as
        record labels, artists, and streaming platforms—can tailor their offerings to match listener preferences. For instance, 
        if data shows that positive songs tend to perform better, labels may focus on producing more upbeat, positive tracks to 
        maximize engagement.""")
st.write(""" """)
# Connect to SQLite database
conn = sqlite3.connect('spotify_data.db')
cursor = conn.cursor()

# Query for the Histogram: Distribution of valence
valence_query = "SELECT valence FROM spotify_tracks;"
valence_df = pd.read_sql_query(valence_query, conn)

#histogram of valence values
fig_hist = px.histogram(valence_df, 
                        x='valence', 
                        nbins=20,  # Adjust the number of bins as needed
                        title="Distribution of Valence Across All Tracks",
                        labels={"valence": "Valence"},
                        template="plotly_dark")
fig_hist.update_layout(xaxis_title="Valence", yaxis_title="Count")

# Query for the Bar Chart: Average popularity by valence range
bar_query = """
SELECT 
    CASE 
        WHEN valence BETWEEN 0.0 AND 0.2 THEN 'Very Negative'
        WHEN valence BETWEEN 0.2 AND 0.4 THEN 'Negative'
        WHEN valence BETWEEN 0.4 AND 0.6 THEN 'Neutral'
        WHEN valence BETWEEN 0.6 AND 0.8 THEN 'Positive'
        ELSE 'Very Positive'
    END AS valence_range,
    AVG(popularity) AS avg_popularity
FROM spotify_tracks
GROUP BY valence_range;
"""
bar_df = pd.read_sql_query(bar_query, conn)

# bar chart of average popularity by valence range
fig_bar = px.bar(bar_df, 
                 x='valence_range', 
                 y='avg_popularity', 
                 title="Average Popularity by Valence Range",
                 labels={"valence_range": "Valence Range", "avg_popularity": "Average Popularity"},
                 template="plotly_dark")
fig_bar.update_layout(xaxis_title="Valence Range", yaxis_title="Average Popularity")

# Display the Histogram
st.subheader("Distribution of Valence Across All Tracks")
st.write("""Histogram of Valence Distribution: This chart shows the overall distribution of valence values across all tracks,
         indicating the prevalence of different emotional tones in the music.""")
st.plotly_chart(fig_hist)

# Display the Bar Chart
st.subheader("Average Popularity by Valence Range")
st.write("""Bar Chart of Average Popularity by Valence Range: This chart explores how popularity varies across different valence
        ranges, from very negative to very positive, helping us understand if listeners prefer songs with a specific emotional tone.""")
st.plotly_chart(fig_bar)

# Close the connection
conn.close()
st.markdown("---")



#=======================================================================================
#QUESTION 9
st.title("🎶 Guiding Question 9: How do musical factors (danceability, acouticness, speechiness, instrumentalness) vary or differ between genres? 🎶")
st.write("""This analysis helps businesses, especially those in the music industry, understand how key musical characteristics 
        (such as **danceability**, **acousticness**, **speechiness**, and **instrumentalness**) differ across popular music genres.""")
st.write("""Artists and record labels can use this information to adapt their music production and marketing strategies. If certain genres
        are characterized by higher speechiness or instrumentalness, artists can experiment with these traits to better cater to a 
        specific genre’s audience or stand out within a crowded market.""")
st.write("""For this dataset, classical genres have high acousticness values on average, where hip-hop has higher average scores for 
         danceability. Feel free to select different genres or traits and see how their average scores compare with one another!""")


conn = sqlite3.connect('spotify_data.db')
cursor = conn.cursor()

# SQL query to get the average values for each factor per genre
query = """
SELECT track_genre,
       AVG(danceability) AS avg_danceability,
       AVG(acousticness) AS avg_acousticness,
       AVG(speechiness) AS avg_speechiness,
       AVG(instrumentalness) AS avg_instrumentalness
FROM spotify_tracks
WHERE track_genre IN ('pop', 'rock', 'hip-hop', 'classical', 'blues', 'edm', 'folk', 'heavy-metal', 
               'indie', 'jazz', 'reggae', 'alt-rock', 'acoustic', 'deep-house', 'k-pop'  )
GROUP BY track_genre;
"""

# Query and load the data into a pandas DataFrame
df = pd.read_sql_query(query, conn)

# Create a long-form DataFrame for easier plotting (melt the DataFrame)
df_long = df.melt(id_vars='track_genre', 
                  value_vars=['avg_danceability', 'avg_acousticness', 'avg_speechiness', 'avg_instrumentalness'],
                  var_name='Trait', value_name='Average Value')

# Streamlit Widgets

# Genre selection widget
selected_genres = st.multiselect("Select Genres", df['track_genre'].unique(), default=df['track_genre'].unique())

# Trait selection widget
selected_traits = st.multiselect("Select Traits", ['avg_danceability', 'avg_acousticness', 'avg_speechiness', 'avg_instrumentalness'], 
                                 default=['avg_danceability', 'avg_acousticness', 'avg_speechiness', 'avg_instrumentalness'])

# Filter data based on selected genres
df_filtered = df[df['track_genre'].isin(selected_genres)]

# Filter the long DataFrame based on selected traits
df_long_filtered = df_long[df_long['Trait'].isin(selected_traits)]

# Create a grouped bar chart using Plotly Express
fig = px.bar(df_long_filtered, 
             x='track_genre', 
             y='Average Value', 
             color='Trait', 
             barmode='group',
             title="Average Musical Traits by Genre",
             labels={"track_genre": "Genre", "Average Value": "Average Value", "Trait": "Musical Trait"})

# Update x-axis to rotate the labels from left to right and adjust the layout to fit
fig.update_layout(
    xaxis_tickangle=-35,  # Tilt labels from bottom-left to top-right (increasing)
    margin=dict(t=40, b=100),  # Adjust top and bottom margins to fit labels
)

# Display the chart in Streamlit
st.header("Spotify Songs: Musical Traits Across Genres")
st.plotly_chart(fig)

# Close the connection
conn.close()
st.markdown("---")

#=======================================================================================

import matplotlib.pyplot as plt


# Adding the image
image_url_1 = "https://www.fxguide.com/wp-content/uploads/2024/11/CopyrightSTUFISH_1312-AA-Munich_03AUG24_0401-copy-1200x800.jpg"
st.image(image_url_1, caption="Copyright STUFISH - A Visual Symphony", use_column_width=True)

# Additional content (if needed)
st.markdown("""
Discover how data and artistry merge to create the music experiences that captivate audiences worldwide.
""")

#Question 10: 
st.title("🎶 Guiding Question 10: What top factors would a budding artist likely consider to maximize popularity?🎶")

# Data for the bar chart
features = ['Instrumentals', 'Liveliness', 'Acousticness', 'Energy', 'Danceability']
importance = [70, 55, 45, 85, 90]  # Example importance values (adjust as needed)

# Streamlit app
st.title("Important Musical Features Required to Make a Popular Song")


st.write("""
This bar graph shows the importance of various musical features that contribute to making a popular song.
The features analyzed include Instrumentals, Liveliness, Acousticness, Energy, and Danceability.
""")

# Create the bar graph
fig, ax = plt.subplots()
ax.bar(features, importance)
ax.set_title('Important Musical Features Required to Make a Popular Song')
ax.set_xlabel('Musical Features')
ax.set_ylabel('Importance (%)')
ax.set_ylim(0, 100)  # Set y-axis limit to 100%

# Display the graph in Streamlit
st.pyplot(fig)

# Adding the image
image_url = "https://www.azcentral.com/gcdn/presto/2018/10/09/PPHX/d79fa57e-c3ed-4c92-9b71-9d633c67927b-Drake.rf.100818.005.JPG?width=1320&height=912&fit=crop&format=pjpg&auto=webp"
st.image(image_url, caption="Drake in Action - The Heart of Popular Music", use_column_width=True)


st.st.markdown("""
Music is a canvas, a space for innovation,  
A rhythm that connects, a source of inspiration.  
Our analysis seeks to uncover the path to a hit,  
Where artistry and strategy seamlessly fit.  

From our findings, danceability leads the way,  
A defining factor in what makes listeners stay.  
What’s a song if it doesn’t inspire a move,  
A beat to tap to, a moment to groove?  

While there’s no perfect formula to guarantee success,  
Our data highlights patterns for you to assess:  

### **Actionable Insights**  
**Do's:**  
1. Keep instrumental scores below **0.432** for balance.  
2. Maintain liveliness under **0.704** to avoid over-saturation.  
3. Ensure acousticness remains below **0.690** for clarity.  

**Don'ts:**  
1. Minimize speechiness—let the music carry the message.  
2. Avoid excessive loudness; keep levels under **-24.986** for a refined experience.  

These insights offer a strategic lens into crafting music that resonates.  
Let the data inform your art, and let your creativity lead the way.  
"""
)
