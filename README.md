# Tutoriel IA Agentique

## Installation

```bash
git clone https://github.com/maximemorge/tutorielIAAgentique.git
cd tutorielIAAgentique
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou sous Windows :
# .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # puis renseignez votre clé API
```

## Structure du projet

```
tutorielIAAgentique/
├── src/
│   └── tutorielIAAgentique/
│       ├── agent.py
│       └── ...
├── tests/
│   └── ...
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Documentation

Pour participer au tutoriel, suivez les instructions disponibles sur le
[wiki du projet](https://github.com/maximemorge/tutorielIAAgentique/wiki).

## Prérequis

- Python 3.11+
- Une clé API Groq (disponible sur [console.groq.com](https://console.groq.com))

En amont du tutoriel, vous pouvez préparer votre machine en suivant les instructions disponibles
[[https://github.com/maximemorge/tutorielIAAgentique/wiki/01Prerequisites][ici]].
