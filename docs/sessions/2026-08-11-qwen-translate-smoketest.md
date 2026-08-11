# Session 2026-08-11 — Smoketest traduction Qwen3-VL (Phase 3)

## Jalon : smoketest traduction texte-only réussi

`benchmarks/qwen_vl_translate_smoketest.py` exécuté avec succès sur 10
vraies lignes de dialogue OCR du corpus Manga109-s (Arisa, déjà observées
dans `data/outputs/demo/demo.txt` en session précédente), vers le français
et l'anglais. Contrairement à `benchmarks/qwen_vl_smoketest.py` (prompt
image+texte, description de scène), ce smoketest utilise un prompt
**texte seul** (pas de `image_url` dans les messages) — le projecteur
vision MTMD n'est pas sollicité.

## Résultats mesurés

| Langue cible | Chargement | Latence min/moy/max | RAM pic |
|---|---|---|---|
| fr | 5.44s | 1.06s / 2.29s / 4.95s | 5220 Mo |
| en | 1.96s | 1.02s / 1.78s / 3.21s | 5233 Mo |

Très en-deçà des 111.62s/image mesurés pour la description de scène
(`docs/sessions/2026-08-10-qwen-smoketest.md`) — attendu, l'essentiel du
coût de ce dernier vient de l'encodage vision, absent ici. Pour un chapitre
d'une trentaine de segments dialogue/thought, la traduction ajoute environ
1 minute au total, négligeable devant les 2-4 minutes déjà observées pour
la seule synthèse Kokoro. Pas de risque de rendre le pipeline impraticable,
pas de besoin de traduction par lot groupé pour l'instant.

## Qualité — limite connue et acceptée pour le MVP

9 traductions sur 10 fidèles au sens d'origine, sur les deux langues. Une
phrase donne un contresens, **identique sur fr et en**, donc pas un hasard
de génération mais une vraie limite du modèle sur cette tournure :

```
JA : 泣くなって！          (= « arrête de pleurer ! »)
FR : « Crie ! »             (sens inversé)
EN : « Cry out! »            (même sens inversé)
```

`泣くなって！` combine une négation (`泣くな` = ne pleure pas) et une
particule de citation/emphase (`って`) — une construction elliptique
typique de l'oral familier dans les bulles de manga. Le modèle interprète
la phrase à l'envers plutôt que de reconnaître la négation citée.

**Décision** : acceptable pour un MVP. La traduction par LLM sur du
japonais familier/elliptique (registre dominant des dialogues de manga)
est un problème ouvert, pas spécifique à Qwen3-VL-4B ni corrigible par un
ajustement de prompt local à ce projet — même les meilleurs modèles de
traduction commettent ce type de contresens sur de l'oral japonais très
elliptique. Le contenu reste globalement fidèle (erreur = exception, pas
la norme) ; documenté dans le code
(`QwenVLBackend.translate()`, `pipeline/translation.py`) pour que ce ne
soit pas découvert par surprise en production.

Rapports complets : `data/outputs/qwen_vl_translate_smoketest_fr.txt`,
`data/outputs/qwen_vl_translate_smoketest_en.txt`.

## Conclusion

Latence et RAM largement praticables — feu vert pour l'intégration
production sans optimisation préalable. Qualité : limite de contresens
occasionnelle documentée et acceptée, pas bloquante.

## Suite : intégration production (même session)

- `TranslationBackend` (interface, `backends/base.py`) +
  `QwenVLBackend.translate()` (texte seul, même `self._llm` que `describe()`)
- `pipeline/translation.py::translate_dialogues()` — traduit en place les
  segments dialogue/thought japonais, en amont de `assemble_audio()`
- Bug trouvé et corrigé en marge : `_detect_lang()` (`audio_assembler.py`)
  retombait sur `"en-us"` en dur pour tout dialogue sans japonais, y
  compris un dialogue traduit en français — aurait donné le mauvais
  phonémiseur espeak à Kokoro. Fix : `default_lang` calculé depuis
  `narration_lang` de l'appelant.
- `scripts/demo.py::run_demo()` : nouvelle passe séquentielle (charge
  Qwen3-VL, traduit, décharge, avant de charger Kokoro) entre la fusion des
  scripts narratifs et `assemble_audio()` — respecte la contrainte "un seul
  modèle lourd en RAM à la fois" déjà en place pour
  Magiv2/manga-ocr/Kokoro/vision.
- Limite non résolue par ce travail : le pool de voix des personnages
  (`_CHARACTER_VOICE_POOL`, `narrative_builder.py`) reste japonais même
  pour un dialogue traduit — corrige la langue/phonémisation, pas le
  timbre de voix. Chantier séparé à prévoir.
