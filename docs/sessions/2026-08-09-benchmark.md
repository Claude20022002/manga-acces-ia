# Session 2026-08-09 — Benchmark corpus complet

## Résultat final

**28/28 pages OK, 0 erreur** — `benchmarks/benchmark_corpus.py` sur
`data/manga_jpg/` (28 pages), pipeline complet Magiv2 → manga-ocr →
narrative_builder → Kokoro, en 3 passes séquentielles (un backend chargé
une seule fois pour tout le corpus, déchargé avant la passe suivante).

```
=== Résumé benchmark corpus ===
Pages : 28 (28 OK, 0 ERREUR)

Sur les pages OK :
  Panels détectés      : 161 (moyenne 5.8/page)
  Textes détectés      : 190 (moyenne 6.8/page)
  Textes hors panel    : 0 (moyenne 0.0/page)
  Segments narratifs   : 351 (moyenne 12.5/page)
  Durée audio totale   : 1633.2s (moyenne 58.3s/page)
  Temps de traitement  : 2857.6s (moyenne 102.1s/page, hors chargement modèles)

Temps total du run (chargements modèles inclus) : 2865.0s
```

CSV : `data/outputs/benchmark_corpus_final.csv`. Audio : `data/outputs/benchmark_final/`.

## Bug découvert : crash Kokoro sur ponctuation pleine chasse

Le premier run complet (28 pages) a produit **12 pages en erreur** sur
`_assemble_audio_loaded`, toutes avec le même message :
`ValueError: need at least one array to concatenate`.

### Cause exacte

L'exception n'est pas levée par notre code mais **à l'intérieur de
`kokoro_onnx`** (`kokoro_onnx/__init__.py:207`, `Kokoro.create()`) :
phonémisation en `lang="en-us"` d'un texte qui ne contient aucun caractère
reconnu par espeak-ng dans ce mode → liste de chunks audio vide → `Kokoro.create()`
tente `np.concatenate([])` en interne et plante **avant même de retourner**
un résultat à l'appelant.

Diagnostic confirmé par traceback isolé :

```
File ".../kokoro_onnx/__init__.py", line 207, in create
    audio = np.concatenate(audio)
ValueError: need at least one array to concatenate
```

Concerné : ponctuation japonaise pleine chasse isolée (`？` `！` `．．．`
`「」`), chiffres pleine chasse isolés (`２０１２`), texte vide ou
whitespace-only. Les équivalents ASCII (`?`, `!`, `...`) fonctionnent, tout
comme le texte japonais "normal" (au moins un kana/kanji présent).

Une première tentative de correction (vérifier `len(samples) == 0` *après*
l'appel à `self._model.create()`) s'est révélée inopérante : l'appel ne
revient jamais dans ce cas, il lève l'exception directement — il fallait
englober l'appel lui-même dans un `try/except`.

### Fix appliqué

**`backends/kokoro_backend.py`** — `KokoroBackend.synthesize()` : l'appel à
`self._model.create()` est encapsulé dans un `try/except ValueError`. En cas
d'échec, retourne 100ms de silence (`np.zeros(int(SAMPLE_RATE * 0.1))`,
avec `SAMPLE_RATE = 24000` importé depuis `kokoro_onnx.config`, confirmé par
lecture du code source) plutôt que de laisser planter la synthèse. Un
`logger.warning` trace le texte fautif à chaque occurrence. Filet de
sécurité permanent, indépendant du pipeline.

**`pipeline/audio_assembler.py`** — `assemble_audio()` : les segments dont
le texte est vide après `.strip()` sont ignorés (`continue`) avant l'appel à
`synthesize()`, avec `logger.warning`. Ne couvre que le texte vide/whitespace
(la ponctuation pleine chasse isolée n'est pas vide après `.strip()` — c'est
le filet Kokoro qui l'attrape). Filtrage volontairement absent de
`narrative_builder.py` : le script narratif doit rester fidèle à ce que le
pipeline a produit, y compris les textes pauvres ; le filtrage appartient à
la couche de rendu, pas à la couche de données.

Effet cosmétique mineur accepté : si les tout premiers segments d'une page
sont vides et donc ignorés, l'insertion de silence entre segments
(indexée sur la position dans `enumerate()`, pas sur le nombre de segments
réellement synthétisés) peut ajouter 300ms de silence de tête avant le
premier audio réel. À corriger si ça pose problème à l'écoute.

### Vérification

Re-benchmark ciblé sur les 12 pages en erreur (dossier temporaire de liens
symboliques, supprimé après usage) : **12/12 OK**. Re-benchmark complet des
28 pages : **28/28 OK**, confirmant l'absence de régression sur les pages
déjà saines.

### 16 warnings absorbés (run final, 28 pages)

Textes non phonémisables rencontrés, silence retourné pour chacun :

```
！？
．．．
．．．．！！
．．．
２０１２
．．．
！！
？
？
．．．．．
．．．
．．
！？
．．．
．．．
．．．
```

## Autres observations

- **0 texte hors panel sur 28 pages** — le filtrage par centre de bbox
  (`_find_panel_for_text`, centre de la bbox texte contenu dans la bbox du
  panel) s'est révélé robuste sur tout le corpus, aucun cas où le centre
  d'un texte détecté par Magiv2 tombe hors de toute case.
- **Temps de chargement des modèles** (cache HuggingFace chaud) :
  Magiv2 3.84s, manga-ocr 2.28s, Kokoro 0.56s. Chargement une seule fois par
  passe sur les 28 pages (pas de rechargement par page), conformément à la
  contrainte mémoire du projet (12 Go RAM, un seul modèle lourd en RAM à la
  fois).

## Prochaines priorités

1. Voix japonaise Kokoro (`lang="ja"`, preset disponible dans
   `voices-v1.0.bin`) pour améliorer la prononciation en attendant
   l'intégration de la traduction Qwen3-VL.
2. Lecteur accessible HTML+ARIA (Phase 4).
3. Surveiller la stabilisation du support Qwen3-VL dans llama.cpp
   ([ggml-org/llama.cpp#16207](https://github.com/ggml-org/llama.cpp/issues/16207))
   avant de reconsidérer son intégration (traduction).
