# Importing libraries
import pandas as pd 
import matplotlib.pyplot as plt
import statsmodels.api as sm
data = pd.read_csv(r"C:\Users\sebas\OneDrive - BI Norwegian Business School (BIEDU)\GitHub\Portfolio\AB_testing\Data\ab_testing.csv")

# Some data info and generating a dummy for conversion
data["Conver_d"] = data["Conversion"].map({"Yes":1, "No":0})
data["treat"] = data["Group"].map({"B": 1, "A": 0})
# Set up outcomes and subgroups which we want to compare
outcomes = ["Page Views", "Time Spent", "Conver_d"]
subgroups = ["Device", "Location"]
results = []

# Loop through subgroups and outcomes, ie. preform interacted regressions 
plt.rcParams["font.family"] = "Times New Roman"
plt.style.use('_classic_test_patch')
for subgroup in subgroups:
    categories = data[subgroup].unique()
    for category in categories:
        data_subset = data[data[subgroup] == category]
        for outcome in outcomes:
            x = sm.add_constant(data_subset["treat"])
            y = data_subset[outcome]
            model = sm.OLS(y,x).fit(cov_type="HC3")
            coef = model.params["treat"]
            conf_int = model.conf_int().loc["treat"]
            results.append({
                "Subgroup": subgroup,
                "Category": category,
                "Outcome": outcome,
                "Coefficient": coef,
                "CI Lower": conf_int[0],
                "CI Upper": conf_int[1]
            })

# Convert to DF for plotting
results_df = pd.DataFrame(results)

# Plotting point estimate + CI by subgroup
fig, axes = plt.subplots(3, 2, figsize=(15,14), sharey=False)

for i, outcome in enumerate(outcomes):
    for j, subgroup in enumerate(subgroups):
        ax = axes[i, j]
        plot_data = results_df[(results_df["Outcome"] == outcome) & (results_df["Subgroup"] == subgroup)]
        ax.errorbar(plot_data["Category"], plot_data["Coefficient"],
            yerr=[plot_data["Coefficient"] - plot_data["CI Lower"], plot_data["CI Upper"] - plot_data["Coefficient"]],
            fmt="o", capsize=5, linestyle="None")
        ax.axhline(0, color="black", linestyle="--")
        ax.set_title(f"{outcome} by {subgroup}")
        ax.set_ylabel("Treatment Effect")
        ax.set_xlabel(subgroup)

plt.tight_layout()
plt.show()

