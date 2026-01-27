---
title: 'MOSAICapp: An Interactive Web Application for Topic Modelling of Phenomenological Reports'
tags:
  - Python
  - topic modelling
  - phenomenology
  - consciousness research
  - BERTopic
  - natural language processing
authors:
  - name: Romy Beauté
    orcid: 0009-0006-4548-5349
    corresponding: true
    affiliation: "1, 2"
affiliations:
  - name: Sussex Centre for Consciousness Science, University of Sussex, UK
    index: 1
  - name: Sussex AI, University of Sussex, UK
    index: 2
  - name: UCL Centre for Consciousness Research, University College London, UK
    index: 3
date: 27 January 2026
bibliography: paper.bib
---

# Summary

MOSAICapp is a web application for topic modelling of phenomenological text data. It implements the MOSAIC pipeline (Mapping Of Subjective Accounts into Interpreted Clusters), which combines BERTopic [@grootendorst2022bertopic] with transformer-based sentence embeddings [@reimers2019sentence] to identify thematic structure in open-ended subjective reports. Users upload a CSV file, configure analysis parameters through a web interface, and obtain interactive visualisations of discovered topics along with downloadable results. The application is deployed on Hugging Face Spaces for browser-based access, and a Python library (`mosaic_core`) is provided for programmatic use and batch processing.

# Statement of Need

Phenomenological research in consciousness science often relies on open-ended subjective reports to capture experiential dimensions that structured instruments may miss. Questionnaires such as the Altered States of Consciousness Rating Scale [@studerus2010psychometric], the Phenomenology of Consciousness Inventory [@pekala1991quantifying], or the Minimal Phenomenal Experience Questionnaire [@gamma2021mpe] provide validated dimensional measures, but their predefined categories can constrain what researchers are able to discover. Experiences that do not fit the existing structure are typically lost or reduced to "other" responses.

Topic modelling offers a way to discover thematic structure directly from text, without imposing categories in advance. Traditional approaches such as Latent Dirichlet Allocation [@blei2003latent] treat documents as bags of words without capturing semantic context—a limitation when analysing phenomenological language, where meaning often depends on subtle contextual relationships. BERTopic [@grootendorst2022bertopic] improves on this by using transformer-based embeddings that capture semantic similarity, combined with density-based clustering (HDBSCAN; @campello2013density; @mcinnes2017hdbscan) and dimensionality reduction (UMAP; @mcinnes2018umap).

However, using BERTopic requires programming expertise, which can be a barrier for researchers interested in studying subjective experience who do not have a background in computer science. MOSAICapp makes this pipeline accessible through a web interface. The application handles text preprocessing and sentence segmentation via NLTK [@bird2009natural], provides configurable UMAP and HDBSCAN parameters, and includes optional LLM-based topic labelling for generating interpretable cluster names.

The tool is intended for consciousness researchers analysing altered states reports, phenomenologists working with first-person experiential descriptions, and qualitative researchers who want to explore their text data computationally without writing code.

# Features

MOSAICapp provides several features relevant to phenomenological research:

**Sentence-level analysis.** Reports can be segmented into individual sentences before embedding, allowing finer-grained topic discovery. This is useful when single reports contain multiple distinct experiential themes.

**Hyperparameter optimisation.** The application supports Bayesian optimisation via Optuna [@akiba2019optuna] to search for UMAP and HDBSCAN parameters that maximise topic coherence, rather than requiring users to manually tune these values.

**Interactive visualisations.** Results include 2D scatter plots of the topic space (with documents coloured by topic assignment), hierarchical clustering dendrograms showing topic relationships, topic size distributions, and per-topic keyword importance charts.

**Reproducibility.** Configuration parameters and random seeds can be exported, allowing analyses to be reproduced or adjusted.

# Research Applications

MOSAICapp has been used to analyse several phenomenological datasets:

**Stroboscopic phenomenology.** Analysis of 898 sentences from the Dreamachine project—a large-scale public study of stroboscopic light stimulation—identified 12 experiential clusters in the High Sensory condition and 7 in the Deep Listening condition. These included themes related to visual hallucinations, autobiographical memory recall, synesthetic experiences, and altered states of consciousness, extending beyond what traditional stroboscopic light research has typically measured [@beaute2025mosaic].

**Minimal Phenomenal Experience.** The pipeline was applied to free-text descriptions of "pure awareness" states collected as part of the MPE study [@gamma2021mpe], enabling data-driven exploration of how participants describe these experiences in their own words.

**Psychedelic phenomenology.** The tool has been used to analyse micro-phenomenological interview transcripts from DMT and 5-MeO-DMT studies, identifying fine-grained experiential clusters in datasets containing several thousand sentences.

# Acknowledgements

This work was funded by the Be.AI Leverhulme doctoral scholarship. I thank Maarten Grootendorst for creating BERTopic, and David Schwartzman, Guillaume Dumas, and Anil Seth for discussions on the MOSAIC methodology. I thank Thomas Metzinger and Christopher Timmermann for sharing their MPE and DMT data, and for facilitating the development of this research.

# References