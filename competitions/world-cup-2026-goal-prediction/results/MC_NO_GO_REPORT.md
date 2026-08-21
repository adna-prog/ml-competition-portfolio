# Dixon-Coles Monte Carlo — corrected practice version

## Correction relative à la première version

La première version n'appliquait pas le correctif Dixon-Coles `tau` lors de
l'échantillonnage des scorelines. Elle utilisait un Poisson indépendant malgré
l'ajustement de `tau` dans la vraisemblance. Son score privé `0.432626991` ne
permet donc pas de rejeter l'architecture entière.

La version corrigée applique `tau` aux matchs de groupes et de knockout, conserve
3 000 simulations et utilise un assignment Hongrois global pour imposer les
quotas 2026.

## Limites restantes

- Les groupes 2026 ne sont pas fournis dans l'archive.
- Les groupes utilisés sont une allocation déterministe documentée.
- Le bracket officiel n'est pas disponible dans les données fournies.
- Cette version reste donc un laboratoire méthodologique, pas une reproduction
  exacte du tournoi.

## Private score check

- First MC implementation: `0.432626991`
- Corrected Dixon-Coles MC (`tau` applied): **`0.447938580`**
- Gain versus first MC: `+0.015311588`
- Gain versus heuristic baseline `0.439492695`: `+0.008445885`
- Gap versus Optuna joint champion `0.543579701`: `-0.095641121`

The correction helped materially, confirming that the missing `tau` application
was a real implementation bug. However, the corrected simulation remains far
below the quota-constrained Optuna model. The artificial 2026 groups and missing
official bracket remain major unresolved limitations.

Decision: retain as an educational simulation component, not as the practice
champion or preferred submission artifact.
