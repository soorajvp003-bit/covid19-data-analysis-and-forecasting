# Preprocessing Summary

- Raw rows loaded: 247
- Cleaned rows saved: 247
- Duplicate rows removed: 0
- Selected columns: iso_code, continent, location, date, population, total_cases, new_cases, total_deaths, new_deaths, people_vaccinated, people_fully_vaccinated, total_boosters, total_vaccinations, new_vaccinations

## Cleaning Notes
- Cumulative columns were forward-filled inside each country to preserve monotonic totals.
- Daily case, death, and vaccination gaps were treated as zero because missing daily values usually mean no new report for that day.
- Recovery trends are represented with an estimated recovery proxy because OWID does not publish official recovery counts in the main file.
- Rows such as World, continents, and income groups are retained but tagged with `is_country = False` for flexible filtering.