---
title: 'MOSAICapp: An Interactive Web Application for Topic Modelling of Phenomenological Reports'
tags:
  - Python
  - topic modelling
  - phenomenology
  - consciousness research
  - BERTopic
  - natural language processing
  - qualitative research
authors:
  - name: Romy Beauté
    orcid: 0009-0006-4548-5349
    corresponding: true
    affiliation: "1, 2"
  - name: Guillaume Dumas
    orcid: 0000-0002-2253-1844
    affiliation: "3, 4, 5, 6"
  - name: David Schwartzman
    orcid: 0000-0002-3779-396X
    affiliation: "1"
  - name: Anil K. Seth
    orcid: 0000-0002-1421-6051
    affiliation: "1, 6"
affiliations:
  - name: Sussex Centre for Consciousness Science, University of Sussex, UK
    index: 1
  - name: Sussex AI, University of Sussex, UK
    index: 2
  - name: CHU Sainte-Justine Azrieli Research Center, Montréal, Québec, Canada
    index: 3
  - name: Department of Psychiatry and Addictology, University of Montréal, Montréal, Québec, Canada
    index: 4
  - name: Mila – Quebec Artificial Intelligence Institute, Montréal, Québec, Canada
    index: 5
  - name: Program for Brain, Mind, and Consciousness, Canadian Institute for Advanced Research, Toronto, Ontario, Canada
    index: 6
date: 18 February 2026
bibliography: paper.bib
---

# Summary

MOSAICapp is a web application for topic modelling of phenomenological text data. It implements the MOSAIC pipeline (Mapping Of Subjective Accounts into Interpreted Clusters) [@beaute2025mosaic], combining BERTopic [@grootendorst2022bertopic] with transformer-based sentence embeddings [@reimers2019sentence] to identify thematic, experiential structure in open-ended subjective reports (experiential topics). Users upload a CSV file, configure analysis parameters through a graphical interface, and obtain interactive visualisations of discovered topics along with downloadable results. The application is deployed on Hugging Face Spaces for browser-based access without installation, can be run locally via Docker or direct Python installation, and a Python library (`mosaic_core`) is provided for programmatic use and batch processing.

# Statement of Need

Phenomenological research in consciousness science often relies on open-ended subjective reports to capture experiential dimensions that structured instruments may miss. Questionnaires such as the Altered States of Consciousness Rating Scale [@studerus2010psychometric] provide validated dimensional measures, but their predefined categories can constrain what researchers discover. Experiences that do not fit the existing structure are typically lost or reduced to "other" responses. Moreover, many studies do collect open-text reports alongside structured measures, yet without accessible tools for systematic analysis, especially for big dataset, these rich qualitative data can often remain unexplored—representing a significant untapped resource for understanding subjective experience.

Qualitative coding of experiential reports typically requires researchers to iteratively read, annotate, and categorise text. This process is time-intensive, scaling poorly to large datasets—a single researcher can systematically code hundreds of reports, but thousands become impractical. It is also difficult to reproduce: different researchers may apply coding schemes inconsistently, and even the same researcher may drift in their interpretations over time, introducing subjective bias that is hard to quantify or control. Topic modelling offers a computational alternative: it discovers thematic structure directly from text without imposing categories in advance, scales to arbitrarily large datasets, and produces reproducible results given identical data and parameters. BERTopic [@grootendorst2022bertopic] is particularly suited for phenomenological language: its transformer-based embeddings capture semantic relationships that depend on subtle contextual cues, while density-based clustering (HDBSCAN; @campello2013density; @mcinnes2017hdbscan) and dimensionality reduction (UMAP; @mcinnes2018umap) identify structure without predefined categories.
Beyond clustering, Large Language Models (LLMs) can generate interpretable topic labels grounded in phenomenological terminology, moving beyond keyword-based representations. MOSAICapp integrates both capabilities into an accessible web interface (Figure @fig:interface1), enabling consciousness researchers, phenomenologists, and qualitative researchers to explore their text data computationally without writing code.

However, using BERTopic and LLM integration requires programming expertise, which presents a barrier for researchers studying subjective experience who do not have a background in computer science. MOSAICapp makes this pipeline accessible through a web interface, enabling consciousness researchers, phenomenologists, and qualitative researchers to explore their text data computationally without writing code.

# State of the Field

Analysing phenomenological text data presents specific challenges that general-purpose topic modelling tools do not address. Single experience reports often contain multiple distinct themes—a participant might describe visual phenomena, emotional responses, and temporal distortions within the same narrative. Report-level analysis conflates these into a single topic assignment, losing thematic granularity. Additionally, researchers need to distinguish between robust inter-subjective patterns (themes shared across many participants) and idiosyncratic accounts (detailed descriptions from individual participants), a distinction that standard topic modelling outputs do not provide.

MOSAICapp addresses these needs through sentence-level tokenisation that preserves fine-grained experiential themes, topic participation metrics that quantify how many unique participants contribute to each topic, and LLM-based labelling with phenomenologically-informed prompts that generate interpretable labels focused on modes of experience rather than content-specific descriptors. These features make the tool particularly suited for consciousness research, where the goal is often to identify invariant structures of experience across individuals.

# Software Design

MOSAICapp is built with Streamlit, chosen for its ability to create interactive web applications from Python scripts while maintaining compatibility with scientific computing libraries (Figure @fig:interface1). The architecture reflects several design decisions driven by the needs of phenomenological research:

**Separation of embedding and clustering.** The pipeline caches embeddings separately from clustering results, enabling rapid parameter exploration. This matters because finding appropriate UMAP and HDBSCAN settings often requires iteration—users can adjust clustering parameters and re-run analysis in seconds without recomputing expensive embeddings.

**Sentence-level analysis.** Unlike typical topic modelling workflows that treat each document as a unit, MOSAICapp supports sentence-level tokenisation. This addresses a specific challenge in phenomenological research: single experience reports often contain multiple distinct themes (e.g., visual phenomena, emotional responses, and temporal distortions in one narrative), and report-level analysis would conflate these into a single topic assignment.

**Topic participation metrics.** Standard topic modelling outputs show topic size and keywords, but not whether a topic reflects shared experience or individual narrative. We added diversity ratio (unique participants / total sentences per topic) because phenomenological research often aims to identify invariant structures across individuals—this metric helps distinguish robust inter-subjective patterns from idiosyncratic accounts (Figure @fig:interface2).

**Transparent LLM prompting.** The system prompt instructs the LLM to perform phenomenological reduction: identifying structural themes rather than content labels, focusing on modes of experience rather than specific objects. The prompts are displayed in the interface so researchers can understand and modify the labelling approach.

**Exposed parameters with guidance.** Rather than hiding complexity behind defaults, the interface exposes UMAP and HDBSCAN parameters with explanations of their effects. This supports methodological transparency and allows researchers to make informed choices about the trade-off between topic granularity and stability.

# Features


![MOSAICapp interface showing analysis of Dreamachine phenomenological reports. The sidebar displays configurable parameters for preprocessing, embedding models, UMAP, and HDBSCAN. The main panel shows the experiential topic map with LLM-generated labels.](MOSAICapp_interface1.png){#fig:interface1}


![Secondary view of the MOSAICapp user interface detailing extended functional parameters, and in particular, topic participation analysis. In this view, each point represents a discovered topic, plotted by total sentences (x-axis) against unique participants contributing to that topic (y-axis). The diagonal line indicates perfect diversity (one sentence per participant). Points near the line represent robust inter-subjective patterns shared across many participants; points falling below indicate topics dominated by detailed individual accounts. Colour encodes diversity ratio (green = high consensus, red = individual-driven)](MOSAICapp_interface2.png){#fig:interface2}

MOSAICapp provides the following capabilities through its web interface:

**Data input and preprocessing.** Users upload CSV files with automatic encoding detection (UTF-8, mac_roman, cp1252, ISO-8859-1), select the text column to analyse, choose between report-level or sentence-level analysis (using NLTK tokenisation; @bird2009natural), filter units shorter than a configurable word threshold, and subsample data for rapid parameter exploration.

**Embedding model selection.** Multiple transformer models are available, including multilingual options, with GPU or CPU processing. The interface links to the MTEB Leaderboard for informed model selection.

**Clustering configuration.** The interface exposes UMAP parameters (n_neighbors, n_components, min_dist), HDBSCAN parameters (min_cluster_size, min_samples), and vectorizer settings (n-gram range, minimum document frequency, standard and custom stopwords). A parameter guide explains the effect of each setting.

**Outlier reduction.** Users can apply BERTopic's outlier reduction strategies (embedding-based or c-TF-IDF-based) with configurable similarity thresholds to reassign unclassified documents.

**LLM-based topic labelling.** Integration with the Hugging Face Inference API generates interpretable topic labels using phenomenologically-informed prompts. The system prompt and user template are displayed for transparency, and labels are cached.

**Quality metrics.** Topic coherence (C_v) computed via Gensim [@rehurek2010gensim] and embedding coherence (average cosine similarity of top words) help evaluate model quality.

**Topic participation analysis.** A diversity ratio (unique participants per topic) and interactive visualisation distinguish shared phenomenological structures from idiosyncratic individual accounts. Users can filter topics by minimum participant count.

**Interactive visualisations.** Results include a 2D scatter plot using DataMapPlot with documents coloured by topic, topic size distributions, and participation charts.

**Export and reproducibility.** Results can be downloaded as CSV (one row per topic or one row per sentence). Run history preserves full configurations for comparison across parameter settings.


# Research Impact Statement

MOSAICapp has been used to analyse several phenomenological datasets, demonstrating its utility for consciousness research:

**Stroboscopic phenomenology.** Analysis of 898 sentences from the Dreamachine project—a large-scale public study of stroboscopic light stimulation—identified 12 experiential clusters in the High Sensory condition and 7 in the Deep Listening condition. These included themes related to visual hallucinations, autobiographical memory recall, synesthetic experiences, and altered states of consciousness, extending beyond what traditional stroboscopic light research has typically measured [@beaute2025mosaic].

**Minimal Phenomenal Experience.** The pipeline was applied to free-text descriptions of "pure awareness" states collected as part of the MPE study [@gamma2021mpe], enabling data-driven exploration of how participants describe these experiences in their own words.

**Psychedelic phenomenology.** The tool has been used to analyse micro-phenomenological interview transcripts from DMT and 5-MeO-DMT studies, identifying fine-grained experiential clusters in datasets containing several thousand sentences.

The Hugging Face Space deployment has received usage from researchers without prior programming experience, demonstrating the tool's accessibility for its intended audience.

# AI Usage Disclosure

Generative AI tools were used during the development of MOSAICapp. GitHub Copilot assisted with code completion and debugging during development of the Streamlit interface. All AI-assisted outputs were reviewed, validated, and edited by the human authors, who made all core design decisions and take full responsibility for the accuracy and originality of the submitted materials. No AI tools were used in the core algorithmic components, which rely on established libraries (BERTopic, UMAP, HDBSCAN, Sentence-Transformers).

# Acknowledgements

This work was funded by the Be.AI Leverhulme doctoral scholarship.

# References
