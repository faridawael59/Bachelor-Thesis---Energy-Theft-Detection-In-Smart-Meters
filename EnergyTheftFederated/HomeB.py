import pandas as pd

df = pd.read_csv("data/HomeB_labeled.csv", parse_dates=["Date & Time"])
print(df.shape)
print(df['anomaly'].value_counts())
print(df.groupby('anomaly').describe().T)  # quick stats per feature

# Check anomaly timestamps distribution
print(df[df['anomaly']==1]['Date & Time'].min(), df[df['anomaly']==1]['Date & Time'].max())
print(df[df['anomaly']==1]['Date & Time'].dt.hour.value_counts().sort_index())
