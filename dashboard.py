import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
sns.set(style='darkgrid')

# Load data
day_df = pd.read_csv("day.csv")
hour_df = pd.read_csv("hour.csv")
day_df['season'] = day_df['season'].replace({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})
day_df['weathersit'] = day_df['weathersit'].replace({
    1: 'Clear and Few clouds',
    2: 'Mist and Cloudy',
    3: 'Light Snow and Light Rain',
    4: 'Heavy Rain and Thunderstorm'
})

hour_df['season'] = hour_df['season'].replace({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})
hour_df['weathersit'] = hour_df['weathersit'].replace({
    1: 'Clear and Few clouds',
    2: 'Mist and Cloudy',
    3: 'Light Snow and Light Rain',
    4: 'Heavy Rain and Thunderstorm'
})
df = day_df.merge(hour_df, on='dteday', how='inner', suffixes=('_d', '_h'))

# Menambahkan definisi day_type
df['day_type'] = np.where(df['weekday_d'] < 5, 'weekday', 'weekend')

# Sidebar
with st.sidebar:
    st.header("Riandra Diva's Dashboard Bike project")
    st.image("bike-logo.png")
    start_date, end_date = st.date_input(
        label='Rentang Waktu', min_value=pd.to_datetime(df["dteday"].min()),
        max_value=pd.to_datetime(df["dteday"].max()),
        value=[pd.to_datetime(df["dteday"].min()), pd.to_datetime(df["dteday"].max())]
    )

# Filter data
main_df = df[(df["dteday"] >= str(start_date)) & (df["dteday"] <= str(end_date))]

# Main header
st.header('Final Project : Bike Dashboard ')
st.subheader('Belajar Analisis Data dengan Python')

# Clustering
features = main_df[['temp_h', 'atemp_h', 'hum_h', 'windspeed_h', 'casual_h', 'registered_h', 'cnt_h']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=4, random_state=42)
main_df['cluster'] = kmeans.fit_predict(features_scaled)

# Plot 1 - Question 1
st.subheader('Graph 1 - Pola Penggunaan Sepeda Terdaftar dan Non-Terdaftar per Jam')

# Hitung total pengguna terdaftar dan tidak terdaftar per jam
hourly_users_total = main_df.groupby('hr')[['registered_h', 'casual_h']].sum()

fig1, ax1 = plt.subplots(figsize=(15, 6))
ax1.plot(hourly_users_total.index, hourly_users_total['registered_h'], label='Registered Users', color='blue')
ax1.plot(hourly_users_total.index, hourly_users_total['casual_h'], label='Casual Users', color='red')
ax1.set_xlabel('Jam')
ax1.set_ylabel('Jumlah Peminjaman Sepeda')
ax1.set_title('Perbandingan Total Peminjaman Sepeda Registered dan Casual per Jam')
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# Question 2: Pola penggunaan sepeda berdasarkan jam, cuaca, dan musimnya
st.subheader('Graph 2 - Pola Penggunaan Sepeda Berdasarkan Jam dalam Sehari')

# Plot 1 - Question 2
fig2, ax2 = plt.subplots(figsize=(12, 8))
pivot_table_counts = pd.pivot_table(main_df, index='hr', columns='season_h', values='cnt_h', aggfunc='sum')
sns.heatmap(pivot_table_counts, cmap='viridis', ax=ax2)
ax2.set_title('Perbandingan Jumlah Peminjaman Sepeda (cnt) pada Jam, Musim, dan Kondisi Cuaca')
ax2.set_xlabel('Musim')
ax2.set_ylabel('Jam (hr)')
st.pyplot(fig2)


# Plot 2
fig2, ax2 = plt.subplots(figsize=(10, 6))
pivot_table = main_df.pivot_table(index='hr', columns='weathersit_h', values='cnt_h', aggfunc='mean')
sns.heatmap(pivot_table, cmap='viridis', ax=ax2)
ax2.set_title('Pola Penggunaan Bike Sharing Berdasarkan Jam dalam Sehari dan Kondisi Cuaca')
ax2.set_xlabel('Kondisi Cuaca')
ax2.set_ylabel('Jam (hr)')
st.pyplot(fig2)


# Question 3: Pola persebaran tingkat peminjaman sepeda pada jam dan musim
st.subheader('Graph 3 - Pola Persebaran Tingkat Peminjaman Sepeda pada Jam dan Musim')
pivot_table_counts = pd.pivot_table(hour_df, index='hr', columns='season', values='cnt', aggfunc='mean')
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(pivot_table_counts, cmap='viridis', ax=ax)
ax.set_title('Perbandingan Jumlah Peminjaman Sepeda (cnt) pada Jam dan Musim')
ax.set_xlabel('Musim')
ax.set_ylabel('Jam (hr)')
st.pyplot(fig)

# Question 4: Pengaruh musim dan hari terhadap peminjaman sepeda pengguna registered dan non-registered
st.subheader('Graph 4 - Pengaruh Musim dan Hari terhadap Peminjaman Sepeda')
season_day_users = main_df.groupby(['season_d', 'day_type']).agg({'registered_h': 'sum', 'casual_h': 'sum'})

# Pilih fitur untuk clustering
features = season_day_users[['registered_h', 'casual_h']]

# Normalisasi data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Gunakan K-Means untuk clustering
kmeans = KMeans(n_clusters=4, random_state=42)
season_day_users['cluster'] = kmeans.fit_predict(features_scaled)

# Visualisasi hasil clustering menggunakan bar plot
fig3, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x='season_d', y='registered_h', hue='day_type', data=season_day_users, palette='Set2', ax=ax)
ax.set_title('Perbandingan Jumlah Peminjam Terdaftar Sepeda pada Weathersit dan Hari')
ax.set_xlabel('Season')
ax.set_ylabel('Jumlah Peminjam Terdaftar')
ax.legend(title='Day Type')
st.pyplot(fig3)


fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x='season_d', y='registered_h', hue='cluster', data=season_day_users, palette='Set2', ax=ax)
ax.set_title('Perbandingan Jumlah Peminjam Terdaftar Sepeda pada Weathersit dan Hari')
ax.set_xlabel('Musim')
ax.set_ylabel('Jumlah Peminjam Terdaftar')
ax.legend(title='Cluster')
st.pyplot(fig)

# Question 5: Pengaruh cuaca terhadap pengguna registered dan non-registered
st.subheader('Graph 5 - Pengaruh Kondisi Cuaca terhadap Pengguna Sepeda')
weather_users = main_df.groupby('weathersit_d')[['registered_h', 'casual_h']].sum()
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='registered_h', y='casual_h', hue='weathersit_d', data=main_df, 
                palette={'Clear and Few clouds': 'red', 'Mist and Cloudy': 'yellow', 
                         'Light Snow and Light Rain': 'blue', 'Heavy Rain and Thunderstorm': 'green'}, ax=ax)
ax.set_title('Dampak Kondisi Cuaca terhadap Pengguna Registered dan Non-Registered')
ax.set_xlabel('Jumlah Pengguna Terdaftar')
ax.set_ylabel('Jumlah Pengguna Non-Terdaftar')
ax.legend(title='Kondisi Cuaca')
st.pyplot(fig)
