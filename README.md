# Task_guided_attention
Investigation of Task-Guided Attention Mechanisms in Vision Models.

Dette repositoriet inneholder en modulær og selvstendig implementasjon av **AbSViT** (*Analysis-by-Synthesis Vision Transformer*), tilpasset for forskning på koble menneskelige signaler (som blikk/gaze og peking) inn i nevrale oppmerksomhetsmekanismer.

Prosjektet bygger på arbeidet til Baifeng Shi et al. (CVPR 2023 / UC Berkeley 2024) og fokuserer på hvordan top-down priorer ($\xi$) kan modulere interne attention-kart for å isolere relevante objekter i sammensatte scener.

---

## 🛠️ Oppsett og installasjon

Prosjektet bruker [uv](https://github.com/astral-sh/uv) for rask og konsistent pakkehåndtering med Python 3.12.

### 1. Klon repositoriet
```bash
git clone <repository-url>
cd Task_guided_attention

```

### 2. Sett opp virituelt miljø
```bash
uv venv --python 3.12
uv sync
```

### 📦 Nedlasting av forhåndstrente vekter (Checkpoints)
Sjekkpunktene fra Berkeley Box-serveren inneholder forhåndstrente vekter og lagres lokalt i checkpoints/-mappen. Mappen ignoreres automatisk av Git via .gitignore.
For å laste ned vektene for AbSViT-Tiny (eller andre varianter), kjør følgende i terminalen fra rotmappen:
```bash
# Opprett mappen for sjekkpunkter
mkdir -p checkpoints

# Last ned AbSViT-Tiny (.pth)
curl -L -o checkpoints/absvit_tiny.pth "[https://berkeley.box.com/shared/static/4z5y181r02b8034r2b8109u8x2c15t14.pth](https://berkeley.box.com/shared/static/4z5y181r02b8034r2b8109u8x2c15t14.pth)"

# (Valgfritt) Last ned AbSViT-Small (.pth)
curl -L -o checkpoints/absvit_small.pth "[https://berkeley.box.com/shared/static/3wpkf5qo31ghb4dzehczup4pfh24xmve.pth](https://berkeley.box.com/shared/static/3wpkf5qo31ghb4dzehczup4pfh24xmve.pth)"
```

### 💻 Kjøring av baseline-test
For å verifisere at modellarkitekturen, avhengighetene og sjekkpunktene fungerer som forventet på din maskin (inkludert Apple Silicon / MPS), kjør testskriptet:

```bash
uv run scripts/test_baseline.py
```

### 📂 Prosjektstruktur

Task_guided_attention/
├── checkpoints/          # Lokale .pth-filer (ignorert av git)
├── models/
│   └── AbSViT/
│       └── absvit.py     # Modellantarkitektur og top-down feedback-logikk
├── scripts/
│   └── test_baseline.py  # Skript for verifisering av baseline på mps/cpu
├── .gitignore            # Ignorerer .venv, checkpoints/ og cacher
├── pyproject.toml        # Prosjektkonfigurasjon og avhengigheter
└── uv.lock               # Låst avhengighetstre for uv