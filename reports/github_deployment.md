# GitHub Deployment Instructions

## 1. Initialize Git

```bash
git init
git add .
git commit -m "Add COVID-19 global analysis and forecasting project"
```

## 2. Create a GitHub Repository

- Open GitHub and create a new repository named `COVID19-Data-Analysis`
- Do not add a README on GitHub because this project already includes one

## 3. Connect Local Project to GitHub

```bash
git remote add origin https://github.com/<your-username>/COVID19-Data-Analysis.git
git branch -M main
git push -u origin main
```

## 4. Recommended Repository Sections

- Add a strong repository description
- Pin the repository on your GitHub profile
- Add topic tags such as `python`, `data-analysis`, `machine-learning`, `streamlit`, `covid19`, `forecasting`

## 5. Improve the Visual Presentation

- Run `main.py` and add screenshots from the `images/` folder to the README
- Publish the Streamlit dashboard
- Add the project link to your resume, portfolio, GitHub pinned section, and LinkedIn

## 6. Optional Streamlit Cloud Deployment

- Push the repository to GitHub
- Open Streamlit Community Cloud
- Deploy using `dashboard/app.py`
- Set Python version and install dependencies from `requirements.txt`
