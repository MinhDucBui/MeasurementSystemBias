from haversine import haversine, Unit
import pandas as pd
from itertools import combinations

def create_combinations(df):
    # Generate all pairs of cities

    location_map = dict(zip(df["city"], df["location"]))

    cities = df['city'].tolist()
    city_pairs = list(combinations(cities, 2))
    df_pairs = pd.DataFrame(city_pairs, columns=['City1', 'City2'])
    df_pairs['Location1'] = df_pairs['City1'].map(location_map)

    df_pairs['Location2'] = df_pairs['City2'].map(location_map)
    return df_pairs

def calc_distance(city1, city2, country):

    distance_km = haversine(city1, city2, unit=Unit.KILOMETERS)
    return distance_km



import pandas as pd
from itertools import combinations
from tqdm import tqdm
tqdm.pandas()


mapping = {
    "Germany": "de.csv",
    "China": "cn.csv",
    "Russia": "ru.csv",
    "Japan": "jp.csv",
    "USA": "us.csv"
}

country_mapping = {
    "Germany": ["German", "DEU"],
    "China":  ["Chinese", "CHN"],
    "Russia": ["Russian", "RUS"],
    "Japan":  ["Japanese", "JPN"],
    "USA":    ["US", "USA"],
}

dfs_metric = []
for country, file in mapping.items():
    df = pd.read_csv(file)
    df['population'] = df['population'].astype(str)
    df['population'] = df['population'].str.replace(',', '')
    df['population'] = df['population'].str.replace(' ', '').astype(float)
    df = df.sort_values(by='population', ascending=False)

    df = df[:80]
    df = df.sample(n=40, random_state=42)  # pick 40 random rows (seeded for reproducibility)

    df["location"] = df.progress_apply(lambda row: (row["lat"], row["lng"]), axis=1)

    # Create a DataFrame to store the pairs
    df = create_combinations(df)

    df["value"] = df.progress_apply(lambda row: calc_distance(row["Location1"], row["Location2"], country), axis=1)
    df["object"] = df.apply(
        lambda row: (
            f"straight-line distance between {country_mapping[country][0]} cities "
            f"{row['City1']} and {row['City2']}"
        ),
        axis=1
    )
    df["unit"] = "Kilometer (km)"
    df["ISO3"] = country_mapping[country][1]
    df["Country"] = country_mapping[country][0]
    df = df[["object", "unit", "ISO3", "value", "City1", "City2", "Country"]]
    if "USA" == country:
        df_us = df
    else:
        dfs_metric.append(df)


df_metric = pd.concat(dfs_metric)
df_us.to_csv(f"us_city_distances.csv", index=False)
df_metric.to_csv(f"metric_city_distances.csv", index=False)
