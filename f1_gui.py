import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

races = pd.read_csv('data/raw/races.csv')
results = pd.read_csv('data/raw/results.csv')
drivers = pd.read_csv('data/raw/drivers.csv')
constructors = pd.read_csv('data/raw/constructors.csv')
sprint_results = pd.read_csv('data/raw/sprint_results.csv')


def get_driver_standings(year, races, results, sprint_results, drivers):
    season_races = races[races['year'] == year]
    season_results = results[results['raceId'].isin(season_races['raceId'])]

    points = season_results.groupby('driverId')['points'].sum().reset_index()

    season_sprint = sprint_results[sprint_results['raceId'].isin(season_races['raceId'])]
    sprint_points = (
        season_sprint.groupby('driverId')['points'].sum()
        .reset_index().rename(columns={'points': 'sprint_points'})
    )

    combined = points.merge(sprint_points, on='driverId', how='left')
    combined['sprint_points'] = combined['sprint_points'].fillna(0)
    combined['total_points'] = combined['points'] + combined['sprint_points']

    combined = combined.merge(drivers[['driverId', 'forename', 'surname']], on='driverId')
    return combined.sort_values('total_points', ascending=False).reset_index(drop=True)


def get_season_points_progression(year, races, results, sprint_results, drivers, top_n=5):
    season_races = races[races['year'] == year].sort_values('round')
    season_results = results[results['raceId'].isin(season_races['raceId'])].copy()
    season_sprint = sprint_results[sprint_results['raceId'].isin(season_races['raceId'])].copy()

    combined_results = pd.concat([
        season_results[['raceId', 'driverId', 'points']],
        season_sprint[['raceId', 'driverId', 'points']]
    ])
    combined_results = combined_results.groupby(['raceId', 'driverId'])['points'].sum().reset_index()
    combined_results = combined_results.merge(season_races[['raceId', 'round']], on='raceId')
    combined_results = combined_results.sort_values(['driverId', 'round'])
    combined_results['cumulative_points'] = combined_results.groupby('driverId')['points'].cumsum()

    final_totals = combined_results.groupby('driverId')['points'].sum().sort_values(ascending=False)
    top_drivers = final_totals.head(top_n).index

    result = combined_results[combined_results['driverId'].isin(top_drivers)]
    result = result.merge(drivers[['driverId', 'forename', 'surname']], on='driverId')
    return result


def get_season_checkpoint(year, checkpoint_fraction, races, results, sprint_results, drivers):
    season_races = races[races['year'] == year].sort_values('round')
    total_races = len(season_races)
    checkpoint_round = max(1, round(total_races * checkpoint_fraction))

    races_so_far = season_races[season_races['round'] <= checkpoint_round]
    remaining_races = total_races - checkpoint_round

    results_so_far = results[results['raceId'].isin(races_so_far['raceId'])]
    sprint_so_far = sprint_results[sprint_results['raceId'].isin(races_so_far['raceId'])]

    combined = pd.concat([
        results_so_far[['driverId', 'points']],
        sprint_so_far[['driverId', 'points']]
    ])

    points_so_far = combined.groupby('driverId')['points'].sum().reset_index()
    points_so_far = points_so_far.rename(columns={'points': 'points_at_checkpoint'})

    leader_points = points_so_far['points_at_checkpoint'].max()
    points_so_far['points_behind_leader'] = leader_points - points_so_far['points_at_checkpoint']
    points_so_far['races_remaining'] = remaining_races
    points_so_far['year'] = year

    all_season_results = results[results['raceId'].isin(season_races['raceId'])]
    all_season_sprint = sprint_results[sprint_results['raceId'].isin(season_races['raceId'])]
    final_combined = pd.concat([
        all_season_results[['driverId', 'points']],
        all_season_sprint[['driverId', 'points']]
    ])
    final_points = final_combined.groupby('driverId')['points'].sum()
    champion_id = final_points.idxmax()

    points_so_far['is_champion'] = (points_so_far['driverId'] == champion_id).astype(int)

    return points_so_far


recent_seasons = [year for year in sorted(races['year'].unique()) if year >= 2010 and year <= 2024]

training_rows = []
for year in recent_seasons:
    try:
        checkpoint_data = get_season_checkpoint(year, 0.75, races, results, sprint_results, drivers)
        training_rows.append(checkpoint_data)
    except Exception:
        pass

training_data = pd.concat(training_rows, ignore_index=True)

feature_columns = ['points_at_checkpoint', 'points_behind_leader', 'races_remaining']
X = training_data[feature_columns]
y = training_data['is_champion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)


window = tk.Tk()
window.title("F1 Championship Explorer")
window.geometry("900x700")

top_frame = tk.Frame(window)
top_frame.pack(pady=10)

tk.Label(top_frame, text="Enter Year:", font=("Arial", 12)).pack(side="left", padx=5)

year_entry = tk.Entry(top_frame, font=("Arial", 12), width=8)
year_entry.pack(side="left", padx=5)
year_entry.insert(0, "2021")

display_frame = tk.Frame(window)
display_frame.pack(fill="both", expand=True, padx=10, pady=10)


def show_standings():
    for widget in display_frame.winfo_children():
        widget.destroy()

    try:
        year = int(year_entry.get())
    except ValueError:
        error_label = tk.Label(display_frame, text="Please enter a valid year (e.g. 2021)", fg="red")
        error_label.pack()
        return

    standings = get_driver_standings(year, races, results, sprint_results, drivers)

    if len(standings) == 0:
        error_label = tk.Label(display_frame, text=f"No data available for {year} in the dataset.", fg="red")
        error_label.pack()
        return

    columns = ("Position", "Driver", "Points")
    tree = ttk.Treeview(display_frame, columns=columns, show="headings", height=15)

    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=150, anchor="center")

    for i, row in standings.iterrows():
        driver_name = f"{row['forename']} {row['surname']}"
        tree.insert("", "end", values=(i + 1, driver_name, row['total_points']))

    tree.pack(fill="both", expand=True)


def show_prediction():
    for widget in display_frame.winfo_children():
        widget.destroy()

    tk.Label(display_frame, text="Enter current top drivers (name and points), one per line, format: Name,Points",
              font=("Arial", 10), fg="gray").pack(pady=5)

    input_box = tk.Text(display_frame, height=8, width=40, font=("Arial", 11))
    input_box.pack(pady=5)
    input_box.insert("1.0", "Antonelli,179\nRussell,154\nHamilton,147\nLeclerc,108\nNorris,97")

    races_left_frame = tk.Frame(display_frame)
    races_left_frame.pack(pady=5)
    tk.Label(races_left_frame, text="Races remaining this season:", font=("Arial", 11)).pack(side="left")
    races_left_entry = tk.Entry(races_left_frame, width=5, font=("Arial", 11))
    races_left_entry.pack(side="left", padx=5)
    races_left_entry.insert(0, "13")

    result_label = tk.Label(display_frame, text="", font=("Arial", 11), justify="left")
    result_label.pack(pady=10)

    def calculate_prediction():
        lines = input_box.get("1.0", "end").strip().split("\n")
        names = []
        points = []
        for line in lines:
            if "," in line:
                name, pts = line.split(",")
                names.append(name.strip())
                points.append(float(pts.strip()))

        try:
            races_left = int(races_left_entry.get())
        except ValueError:
            result_label.config(text="Please enter a valid number for races remaining.", fg="red")
            return

        leader = max(points)
        behind = [leader - p for p in points]

        input_df = pd.DataFrame({
            'points_at_checkpoint': points,
            'points_behind_leader': behind,
            'races_remaining': [races_left] * len(points)
        })

        probabilities = model.predict_proba(input_df)[:, 1]

        result_text = "Championship Probability:\n\n"
        for name, prob in sorted(zip(names, probabilities), key=lambda x: -x[1]):
            result_text += f"{name}: {prob * 100:.1f}%\n"

        result_label.config(text=result_text, fg="black")

    calc_button = tk.Button(display_frame, text="Calculate Prediction", font=("Arial", 11), command=calculate_prediction)
    calc_button.pack(pady=5)


button_frame = tk.Frame(window)
button_frame.pack(pady=5)

standings_button = tk.Button(button_frame, text="Show Standings", font=("Arial", 11), command=show_standings)
standings_button.pack(side="left", padx=10)

prediction_button = tk.Button(button_frame, text="Predict Championship", font=("Arial", 11), command=show_prediction)
prediction_button.pack(side="left", padx=10)

window.mainloop()