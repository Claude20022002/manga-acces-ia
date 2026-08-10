# Session 2026-08-10 — Smoketest Qwen3-VL

## Jalon : premier smoketest Qwen3-VL réussi

`benchmarks/qwen_vl_smoketest.py` exécuté avec succès sur
`data/manga_jpg/1-1.jpg`, via `MTMDChatHandler` (llama-cpp-python) sur
`Qwen3VL-4B-Instruct-Q4_K_M.gguf` + `mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf`.
Le chemin générique documenté en Phase 5 (section "État Qwen3-VL") fonctionne
tel quel sur ce CPU, sans rencontrer le bug connu
abetlen/llama-cpp-python#2098 (`AttributeError` sur `sampler`) — l'échec
anticipé au niveau des bindings Python ne s'est pas produit sur cette
combinaison de versions.

## Résultats mesurés

- Temps de chargement : 2.12s
- RAM après chargement : 4722 Mo
- Temps d'inférence : 111.62s/image
- RAM pic : 7110 Mo

## Sortie qualitative (`data/manga_jpg/1-1.jpg`)

> Dans une scène d'action sous la pluie battante, un personnage en tenue de
> samouraï est projeté en l'air, les bras tendus, alors que des
> éclaboussures d'eau et des feuilles secouées par le vent soulignent la
> violence du temps. L'atmosphère est intense, avec des montagnes lointaines
> et une végétation dense qui s'agite sous la pluie, créant une ambiance de
> chaos naturel et de tension.

## Conclusion

Fonctionnel. Deux contraintes actées pour l'intégration :
- **Traitement par lot hors ligne obligatoire** : 111.62s/image interdit
  tout usage interactif/synchrone — la génération de description Qwen3-VL
  doit se faire en batch, en amont de la lecture.
- **Chargement séquentiel obligatoire (RAM)** : pic à 7110 Mo pour Qwen3-VL
  seul, sur une machine à 12 Go — cohérent avec la contrainte déjà en place
  pour Magiv2/manga-ocr/Kokoro (un seul modèle en mémoire à la fois),
  aucune marge pour un chargement concurrent avec un autre backend.

## Prochaine étape

`QwenVLBackend` (implémentation concrète de l'interface backend, cf.
`backends/base.py`) puis intégration dans `pipeline/page_processor.py`.
