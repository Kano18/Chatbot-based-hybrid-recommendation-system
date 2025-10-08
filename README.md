# LLM recommendation system

Build hyrbid recommendation system while integrating with LLMs

## Quickstart

```bash
# Create project from template
cookiecutter path/to/this/template

# cd into project
cd llm-recommendation-system

# Create env (optional) and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Launch Jupyter
jupyter lab
```

## Structure

```
llm-recommendation-system/
├── data/                      # put your datasets here (default: /LLM-reco-sys/csv)
├── notebooks/
│   └── 00_dashboard_starter.ipynb
├── src/llm-recommendation-system/
│   ├── __init__.py
│   └── utils.py
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Notes

- This starter targets **Jupyter notebooks** using **panel** stack (Panel/Plotly/Seaborn).
- Replace sample paths with your real dataset names (default is /LLM-reco-sys/csv).
