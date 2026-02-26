# Scénarios de Test Interactifs (Français)

Ces scénarios sont conçus pour tester le comportement de l'Agent RAG directement en français, afin d'isoler les problèmes liés à la traduction multi-langues et de vérifier si l'architecture RAG et les "Guardrails" fonctionnent correctement dans la langue native des documents.

## Scénario 1 : Bruit Conversationnel et "Anchoring" (Mất thẻ trên tàu)

**Objectif :** Vérifier si l'agent se laisse distraire par le "bruit" (RER B, portefeuille) ou s'il reste concentré sur le document administratif perdu (Titre de séjour). Vérifier aussi le système de "Groundedness Fallback".

**Tour 1 :**
*   **User :** J'ai oublié mon portefeuille dans le RER B ce matin. Mon titre de séjour était dedans. Qu'est-ce que je dois faire maintenant ?
*   **Attentes :** L'agent **NE DOIT PAS** donner les consignes des objets trouvés de la SNCF/RATP. Il doit se concentrer sur la perte du titre de séjour. Il peut soit conseiller de déclarer la perte au commissariat, soit demander (via `[DEMANDER]`) quel est le type exact du titre de séjour perdu pour donner la procédure de duplicata.

## Scénario 2 : Procédure Complexe par Étapes (Naturalisation)

**Objectif :** Tester la capacité de l'agent à gérer une procédure nécessitant de multiples conditions préalables sans "halluciner" de fausses règles.

**Tour 1 :**
*   **User :** Je voudrais faire une demande de naturalisation par décret. Quelles sont les conditions ?
*   **Attentes :** L'agent ne doit pas recracher un mur de texte. Il doit utiliser la logique de `[DEMANDER]` pour clarifier la situation de l'utilisateur (ex: l'âge, la durée de résidence en France de 5 ans, la situation professionnelle) avant de valider si la personne est éligible.

## Scénario 3 : Enquête Juridique avec Conditions (Tiền cọc thuê nhà)

**Objectif :** Vérifier que l'agent de recherche légale pose les bonnes questions lorsque la loi dépend d'une variable non fournie par l'utilisateur.

**Tour 1 :**
*   **User :** Que dit la loi sur la location en France ? Quel est le montant maximum pour le dépôt de garantie (la caution) ?
*   **Attentes :** L'agent doit reconnaître qu'il manque une variable critique. La loi diffère selon le type de bail. L'agent doit `[DEMANDER]` : "Votre logement est-il loué vide (non meublé) ou meublé ?"

**Tour 2 :**
*   **User :** C'est pour une location vide.
*   **Attentes :** L'agent répond avec la loi exacte : Le dépôt de garantie est limité à 1 mois de loyer hors charges pour une location vide (avec citation de service-public.fr).

## Scénario 4 : Hors Sujet Explicite (Quán Phở)

**Objectif :** Vérifier que le Guardrail de détection de sujet (Topic Guardrail) fonctionne bien en français.

**Tour 1 :**
*   **User :** Tu connais un bon resto pour manger un Phở dans le 13ème arrondissement de Paris ?
*   **Attentes :** Rejet immédiat par le guardrail. L'agent doit s'excuser et expliquer de manière polie qu'il ne traite que des questions administratives et juridiques françaises.
