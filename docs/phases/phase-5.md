# Phase 5 — Raffinement heuristique, timeline et validation utilisateurs

## Objectif
Améliorer la qualité perçue de l'audio généré sans introduire de nouveau
modèle ni de coût : classification dialogue/sfx par heuristique texte,
timeline JSON pour la navigation par segment dans le lecteur, état des
lieux documenté sur l'intégration Qwen3-VL, et protocole de test avec des
utilisateurs aveugles/malvoyants. Aucune tâche de cette phase n'entraîne
de modèle, n'ajoute de dépendance lourde, ni ne modifie le contrat
MangaPage JSON v1.

## Tâches

### 5.1 Classification dialogue/sfx par heuristique

**Fichier modifié** : `src/manga_access/backends/manga_ocr_backend.py`

Fonction pure ajoutée avant la classe `MangaOCRBackend` :

```python
_KATAKANA_PATTERN = re.compile("[゠-ヿ]")
_JAPANESE_KANA_KANJI_PATTERN = re.compile("[぀-ゟ゠-ヿ一-鿿]")
_SENTENCE_PUNCTUATION = ("。", "、", "？")

_PARTICLES = (
    "は", "が", "を", "に", "で", "と", "も", "の",
    "から", "まで", "より", "へ", "です", "ます", "ました",
    "だった", "ない", "ね", "よ", "わ", "な", "か", "けど", "けれど", "し",
)

_SFX_MAX_LENGTH = 8
_SFX_MIN_KATAKANA_RATIO = 0.7


def _classify_text_type(text: str) -> Literal["dialogue", "sfx"]:
    """Classe un texte OCR en 'dialogue' ou 'sfx' par heuristique texte pure.

    Ne distingue PAS narration/thought (nécessite le contexte panel/bulle
    de Magiv2, non disponible à ce niveau) — tout ce qui n'est pas
    reconnu comme onomatopée reste 'dialogue', comme avant cette heuristique.

    Règle : un texte est classé 'sfx' seulement s'il est court (<= 8
    caractères), sans ponctuation de fin de phrase (。、？), sans aucune
    particule grammaticale japonaise connue, ET majoritairement composé
    de katakana (>= 70% des caractères japonais du texte — tolérance
    conçue pour absorber le bruit OCR observé sur le corpus, ex.
    バタバタッ mal reconnu バタいタッ garde un ratio katakana suffisant).
    """
    stripped = text.strip()
    if not stripped:
        return "dialogue"
    if len(stripped) > _SFX_MAX_LENGTH:
        return "dialogue"
    if any(mark in stripped for mark in _SENTENCE_PUNCTUATION):
        return "dialogue"
    if any(particle in stripped for particle in _PARTICLES):
        return "dialogue"

    japanese_chars = _JAPANESE_KANA_KANJI_PATTERN.findall(stripped)
    if not japanese_chars:
        return "dialogue"

    katakana_ratio = len(_KATAKANA_PATTERN.findall(stripped)) / len(japanese_chars)
    if katakana_ratio >= _SFX_MIN_KATAKANA_RATIO:
        return "sfx"
    return "dialogue"
```

Branchement dans `recognize()` (remplace `type="dialogue"` en dur) :

```python
text = self._model(cropped)
return TextElement(
    id=f"text-{uuid.uuid4().hex[:8]}",
    type=_classify_text_type(text),
    bbox=bbox,
    text_original=text,
    confidence=1.0,
)
```

Aucun changement de schéma requis : `_classify_text_type` retourne un
sous-ensemble (`Literal["dialogue", "sfx"]`) du `Literal` déjà défini sur
`TextElement.type` dans `schemas/manga_page.py`. `narrative_builder.py`
n'a besoin d'aucune modification : `kind=element.type` et le formatage
`[texte]` pour `sfx` s'appliquent automatiquement dès que `type` est
correctement classé en amont.

**Nouveau fichier de test** : `tests/test_manga_ocr_backend.py` (fonctions
plates, docstring française par test, pas de `unittest.mock`, import direct
de `_classify_text_type` — mirroir de la façon dont `_detect_lang` est
testée dans `test_audio_assembler.py`).

Cas de test (entrée → type attendu) :

| Texte | Attendu | Justification |
|---|---|---|
| `"これは重要です"` | dialogue | particules は + です |
| `"バタバタッ"` | sfx | cas corpus connu, katakana pur |
| `"バタいタッ"` | sfx | variante corrompue OCR, ratio katakana suffisant |
| `"ポッ"` | sfx | cas corpus connu |
| `"ドン"` | sfx | onomatopée générique courte |
| `""` / `"   "` | dialogue | défaut sûr |
| `"え？"` | dialogue | ponctuation de phrase |
| `"時は流れた。"` | dialogue | particule は + 。 |
| `"深夜"` | dialogue | kanji seul, ratio katakana = 0 |
| `"Hello there"` | dialogue | aucun caractère japonais |
| `"わあわあ"` | dialogue | limite connue : onomatopée en hiragana — faux négatif assumé |
| `"オーケー"` | sfx | limite connue : emprunt katakana sans particule — faux positif assumé |

Les deux derniers cas sont documentés comme limites connues de
l'heuristique, pas des bugs.

**Fichiers impactés**
- `src/manga_access/backends/manga_ocr_backend.py` (modifié)
- `tests/test_manga_ocr_backend.py` (nouveau)

**Critères de succès**
- `pytest tests/test_manga_ocr_backend.py -v` : tous les cas du tableau passent
- `pytest tests/` : suite existante toujours verte (aucune régression)
- `ruff check src/manga_access/backends/manga_ocr_backend.py` sans erreur
- Vérification manuelle sur le corpus de benchmark (28 pages) : バタバタッ et
  ポッ passent de `[DIALOGUE]` à `[SFX]` dans les transcripts régénérés

**Ce qui n'est PAS dans le périmètre**
- Classification narration/thought (nécessite les métadonnées Magiv2
  `is_essential_text`/contexte de bulle, non transmises à
  `recognize(image, bbox)` — changerait la signature du backend)
- Tout modèle de classification appris (ML) — reste heuristique pure Python
- Correction des SFX flottants non détectés par Magiv2 (limite de Magiv2
  lui-même, déjà documentée dans `docs/sessions/2026-08-09-final.md`)
- Ajustement fin des seuils au-delà des valeurs choisies ici

---

### 5.2 Timeline JSON pour navigation par segment

**Nouveau fichier** : `src/manga_access/schemas/timeline.py`

```python
class TimelineSegment(BaseModel):
    id: str
    kind: Literal["dialogue", "narration", "sfx", "thought"]
    text: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

class Timeline(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    source: dict[str, Any]
    segments: list[TimelineSegment] = Field(default_factory=list)
    # to_json() / from_json(), même pattern que MangaPage/NarrativeScript
```

Pourquoi un modèle Pydantic formel plutôt qu'un dict brut : `MangaPage` et
`NarrativeScript` suivent tous deux ce pattern (`to_json`/`from_json`,
validation des bornes) — le suivre pour `Timeline` donne la validation
gratuite et une convention cohérente dans `schemas/`. `kind` exclut
volontairement `"scene_description"` : ces segments n'ont jamais
d'intervalle audio (filtrés avant synthèse), cohérent avec `save_transcript()`
qui les ignore déjà.

**Fichier modifié** : `src/manga_access/pipeline/audio_assembler.py`

`assemble_audio(...) -> None` devient `assemble_audio(...) -> Timeline`.
Aucun appelant externe ne dépend du type de retour actuel (vérifié par
recherche dans le repo : seuls les tests l'appellent sans utiliser la
valeur de retour ; `benchmarks/benchmark_corpus.py` a sa propre copie
`_assemble_audio_loaded`, indépendante).

Décision explicite sur la sémantique `start_ms`/`end_ms` : **`start_ms` est
le point où la parole commence réellement**, c'est-à-dire calculé sur
`len(combined)` (état réel du flux pydub déjà assemblé) et non sur l'index
brut de boucle. Effet de bord positif : corrige au passage un bug latent où
le code actuel (`if index > 0: combined += silence`) insère 300ms de
silence en tête de fichier si le tout premier segment réel n'est pas à
l'index 0 (ex. un `scene_description` initial, filtré, fait que le premier
dialogue réel est à l'index 1 → silence injecté à tort avant lui). Passer
à `if len(combined) > 0` élimine ce cas sans changer le comportement des
tests existants (aucun d'eux n'a de segment ignoré en tête).

```python
def assemble_audio(script: NarrativeScript, tts_backend: TTSBackend, output_path: Path) -> Timeline:
    ...
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=_SILENCE_BETWEEN_SEGMENTS_MS)
    timeline_segments: list[TimelineSegment] = []

    for segment in script.segments:
        text_stripped = segment.text.strip()
        if not text_stripped:
            continue
        if segment.kind == "scene_description":
            continue

        lang = _detect_lang(text_stripped, segment.kind)
        audio_bytes = tts_backend.synthesize(text_stripped, segment.voice_id, lang=lang)
        audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))

        if len(combined) > 0:
            combined += silence
        start_ms = len(combined)
        combined += audio
        end_ms = len(combined)

        timeline_segments.append(TimelineSegment(
            id=segment.id, kind=segment.kind, text=text_stripped,
            start_ms=start_ms, end_ms=end_ms,
        ))

    tts_backend.unload()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(output_path, format="opus")
    return Timeline(source=script.source, segments=timeline_segments)


def save_timeline(timeline: Timeline, output_path: Path) -> None:
    """Sauvegarde la timeline JSON, sibling du .opus produit par assemble_audio()."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(timeline.to_json(), encoding="utf-8")
```

Design retenu : `assemble_audio` calcule et retourne la `Timeline` (seule
elle dispose des timings de synthèse) ; `save_timeline` ne fait que la
sérialiser — mirroir exact du couple `assemble_audio`/`save_transcript`
existant. Convention de nommage sibling recommandée : `.timeline.json`
(`Path("page.opus").with_suffix(".timeline.json")` → `page.timeline.json`,
pas de collision avec `.txt` du transcript).

**Optionnel, recommandé** : synchroniser `benchmarks/benchmark_corpus.py`
(`_assemble_audio_loaded`) avec le même correctif et produire un fichier
timeline par page benchmarkée (`audio_dir / "timelines" / f"{stem}.timeline.json"`),
pour garder le benchmark représentatif.

**Tests** (ajoutés à `tests/test_audio_assembler.py`) :
- `test_assemble_audio_returns_timeline_with_correct_boundaries`
- `test_timeline_excludes_scene_description_and_empty_segments`
- `test_timeline_no_leading_silence_after_skipped_segment` (non-régression du bug corrigé)
- `test_save_timeline_writes_json` (écrit puis relit via `Timeline.from_json`)
- `test_timeline_roundtrip_json` (mirroir du test roundtrip de `test_narrative.py`)

**`player.html` — changements concrets**

Markup ajouté dans `#player`, après le `.file-picker` existant :
```html
<div class="file-picker">
  <label for="timeline-input">Charger la timeline (.json, optionnel)</label>
  <input type="file" id="timeline-input" accept=".json,application/json">
</div>
<p id="segment-display" class="segment-display" aria-live="polite">Aucune timeline chargée.</p>
```

JS (dans l'IIFE existante, même style que `formatTime`/`updateTimeDisplay`/`setStatus`) :
- État : `timelineSegments = []`, `currentSegmentIndex = -1`
- Handler `timeline-input` `change` : lecture + `JSON.parse`, dégradation
  gracieuse si invalide (`setStatus("Erreur : fichier timeline invalide.")`,
  le lecteur audio reste utilisable sans timeline — pas de régression Phase 4)
- Réinitialisation de `timelineSegments`/`#segment-display` dans le handler
  `fileInput` `change` existant (évite qu'une timeline reste affichée pour
  un autre fichier audio)
- `findSegmentIndex(currentMs)` : essaie d'abord l'index courant (cas
  majoritaire en lecture linéaire), sinon scan linéaire complet (fallback
  après un `seek` arrière — volume de segments par planche trop faible
  pour que le coût soit sensible)
- `updateCurrentSegment()` appelée depuis le handler `timeupdate` natif
  déjà présent (après la mise à jour de `seekBar`), met à jour
  `#segment-display` au format `[KIND] texte` (cohérent avec le préfixe
  de `save_transcript`) seulement si l'index a changé (évite le spam
  `aria-live`)
- Pas de `fetch` réseau — le lecteur reste strictement offline (chargement
  local via `<input type="file">`, comme l'audio)

**Fichiers impactés**
- `src/manga_access/schemas/timeline.py` (nouveau)
- `src/manga_access/pipeline/audio_assembler.py` (modifié)
- `src/manga_access/player.html` (modifié)
- `tests/test_audio_assembler.py` (étendu)
- `benchmarks/benchmark_corpus.py` (optionnel, recommandé)

**Critères de succès**
- `pytest tests/test_audio_assembler.py -v` : existants + nouveaux tests passent
- `pytest tests/` : aucune régression
- `ruff check src/ tests/` et `mypy src/manga_access/pipeline/audio_assembler.py src/manga_access/schemas/timeline.py` sans erreur
- Vérification manuelle : générer audio + timeline d'une planche du corpus,
  ouvrir `player.html`, charger `.opus` puis `.timeline.json`, vérifier au
  clavier/à l'oreille que `#segment-display` se met à jour en cohérence
- Vérification manuelle : `player.html` sans timeline chargée reste
  pleinement fonctionnel (pas de régression Phase 4)

**Ce qui n'est PAS dans le périmètre**
- Rendu visuel riche de la liste des segments avec scroll synchronisé
- Navigation clavier "segment suivant/précédent" dédiée
- Timeline multi-planches / navigation entre planches
- Serveur FastAPI servant la timeline (le player reste 100% offline)

---

### 5.3 État Qwen3-VL / llama.cpp — décision d'intégration

Tâche de vérification et de décision documentée, sans code.

**Constat vérifié (recherche web)** : le support mainline llama.cpp pour
Qwen3-VL (dense + MoE, IMROPE rope vision, deepstack layers) est arrivé via
la PR ggml-org/llama.cpp#16780 et est disponible. Un GGUF **officiel**
`Qwen/Qwen3-VL-4B-Instruct-GGUF` existe sur Hugging Face, plus des
requantisations communautaires (bartowski, unsloth, NexaAI Q4_K_M ~2.5 Go)
— taille compatible avec le budget RAM 12 Go en chargement séquentiel
(cohérent avec le fonctionnement déjà en place pour Magiv2/manga-ocr/Kokoro).
Deux bugs ouverts connus, non bloquants pour ce projet : issue #17200 (cache
KV sur `llama-server` avec requêtes multimodales consécutives — non
pertinent, pas de serveur multi-requêtes ici) ; issue #16895
(sous-utilisation GPU — non pertinent, projet strictement CPU).

**Décision recommandée : planifier l'intégration en Phase 6 (pas en Phase 5,
mais pas un report indéfini)**. Justification :
- GGUF officiel disponible à la bonne taille, budget $0 respecté
- Aucun bug CPU bloquant identifié pour une inférence mono-requête
- La Phase 5 porte déjà 3 tâches concrètes (5.1, 5.2, 5.4) — ajouter
  l'intégration complète (dépendance `llama-cpp-python`, nouveau backend,
  wiring pipeline, prompt design, benchmark RAM/latence) romprait la règle
  `CLAUDE.md` de ne pas écrire de code de production hors du scope de la
  phase courante
- Un report *sans date* contredirait le constat : il n'y a plus de blocage
  technique identifié, seulement un risque de sur-extension du scope de
  cette phase-ci

**Étape de validation minimale recommandée pour Phase 6** (documentée ici,
pas exécutée en Phase 5) : script `benchmarks/qwen_vl_smoketest.py` (statut
script, comme `benchmarks/baseline_naive.py`), chargeant le GGUF Q4_K_M via
`llama-cpp-python`, inférence sur une planche réelle du corpus existant,
mesure du temps de chargement, RAM pic, temps d'inférence, qualité
qualitative — à valider avant tout backend de production.

**Fichiers impactés**
- `docs/phases/phase-5.md` (cette section, texte uniquement)
- Aucun fichier `src/` ou `tests/`

**Critères de succès**
- La section énonce explicitement le constat, la décision (planifier en
  Phase 6, pas reporter indéfiniment) et la justification
- Aucune dépendance `llama-cpp-python` ajoutée à `pyproject.toml` en Phase 5

**Ce qui n'est PAS dans le périmètre**
- Ajout de la dépendance `llama-cpp-python`
- Tout backend `QwenVLBackend` ou équivalent
- Le script `qwen_vl_smoketest.py` lui-même (recommandé pour Phase 6 seulement)
- Wiring de Qwen3-VL dans `scene_descriptor.py`/`page_processor.py`
- Benchmark RAM/latence réel

---

### 5.4 Protocole de test utilisateurs

Tâche de documentation pure, aucun code.

**Emplacement** : `docs/protocols/test-utilisateurs.md`.

**Plan du contenu** :
1. Objectif : hypothèses à valider (navigabilité clavier de l'audio+timeline,
   compréhension du contenu en audio seul, distinguabilité dialogue/sfx à
   l'oreille suite à 5.1)
2. Prérequis bloquant avant tout recrutement : vérifier la compatibilité de
   `player.html` avec NVDA/JAWS/VoiceOver/Orca en interne d'abord
3. Recrutement : via associations (ex. Valentin Haüy, CFPSAA ou équivalents
   locaux), diversité de lecteurs d'écran/plateformes
4. Consentement et éthique : formulaire lui-même accessible, droit de
   retrait à tout moment, anonymisation
5. Environnement : matériel/logiciel d'assistance propre au participant
   (validité écologique), test à distance envisageable
6. Scénarios de tâches concrets sur `player.html` avec audio réel généré
   par le pipeline (charger un fichier, lecture/pause clavier, navigation
   ±5s/10s et vitesse, distinguer un SFX d'un dialogue via la timeline,
   résumer oralement le contenu de la planche)
7. Mesures, via moyens eux-mêmes accessibles : taux de réussite, temps,
   erreurs de navigation, entretien semi-directif enregistré (pas de
   questionnaire visuel), exactitude du résumé de contenu
8. Restitution : compte-rendu texte accessible, alternative textuelle à
   tout tableau/graphique
9. Limites : échantillon restreint (budget $0), retour qualitatif
   prioritaire sur la significativité statistique

**Fichiers impactés**
- `docs/protocols/test-utilisateurs.md` (nouveau)

**Critères de succès**
- Le document existe et couvre les 9 points ci-dessus
- Format cohérent avec les autres documents `docs/`

**Ce qui n'est PAS dans le périmètre**
- Exécution réelle des sessions de test (ce document livre le protocole)
- Recrutement effectif des participants
- Correction de `player.html` suite à un futur retour terrain
- Outillage logiciel pour le consentement accessible

## Critères de succès (phase entière)
- `pytest tests/` : suite complète verte (existants + nouveaux 5.1/5.2)
- `ruff check src/ tests/` et `mypy src/` sans erreur sur les fichiers touchés
- Corpus de benchmark (28 pages) : バタバタッ et ポッ reclassés `[SFX]`
- `player.html` : lecture d'un `.opus` + `.timeline.json` réels avec
  segment courant synchronisé, et fonctionnement inchangé sans timeline
- Décision Qwen3-VL explicite et justifiée dans le document
- Protocole de test utilisateurs livré et complet

## Ce qui n'est PAS dans cette phase
- Intégration effective de Qwen3-VL (backend, prompt, wiring) — Phase 6
- Classification narration/thought par contexte Magiv2
- Détection des SFX flottants hors bulles (limite de Magiv2 lui-même)
- Exécution des sessions de test utilisateurs et analyse de résultats
- Backend FastAPI + SQLite (toujours pas construit)
- Traduction multi-langue des dialogues
