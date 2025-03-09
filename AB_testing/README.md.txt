# A/B Test: Website Background Color

This project explores whether changing a website’s background color from white (**control**) to black (**treatment**) impacts user engagement and conversion.

## Key Insights
- **Conversions**: The black background significantly increases conversion rates.
- **Page Views & Time Spent**: No notable difference between control and treatment.

## How It Works
- **Data**: A synthetic dataset with columns for Group (A/B), Page Views, Time Spent, Conversion (Yes/No), Device, and Location.
- **Analysis**: Regressions (OLS and logistic) test differences in each metric; subgroup analyses consider Device and Location.

## Usage
1. Clone or download the repository.
2. Install dependencies (Python, `pandas`, `statsmodels`, `matplotlib`).
3. Run the analysis script or notebook to see regression results and plots.

## License & Contact
This project is under the MIT License. If you have feedback or suggestions, feel free to open an issue or submit a PR.

