# Development of a Yoruba Linguistic Knowledge Retrieval and Cultural Preservation System Using Retrieval-Augmented Generation (RAG)

## Overview
This repository contains the full implementation of a Yoruba Linguistic Knowledge Retrieval and Cultural Preservation System developed using Retrieval-Augmented Generation (RAG). The system is designed to support culturally grounded information retrieval and natural language generation for the Yoruba language, a low-resource African language with rich tonal, morphological, and cultural structures.

The project integrates heterogeneous Yoruba textual resources and evaluates multiple retrieval strategies and RAG architectures across thematic domains including current affairs, religion, culture, entertainment, and social life.

This repository accompanies the author’s postgraduate thesis submitted to the University of Ibadan.

---

## Research Objectives
The objectives of this research are to:

-	To identify key linguistic features of Yorùbá language such as tone patterns, word structure, and cultural expressions through data analysis, and use these insights to build a more accurate and culturally aware RAG system.  
- To evaluate the impact of various retrieval techniques such as dense retrieval, sparse retrieval, hybrid approaches across the domains of current affairs, entertainment, religion, culture, and social life.  
-	To optimise a Retrieval-Augmented Generation (RAG) system, including Naive, Advanced, and Modular variants, for enhancing text-based knowledge retrieval in the Yoruba language across the domains of current affairs, entertainment, religion, culture, and social life.  
-	To investigate the influence of different large language models on the generation quality and cultural appropriateness of Yoruba text outputs across Naive, Advanced, and Modular RAG systems.  
-	To assess the accuracy and relevance of RAG-generated outputs in Yoruba using quantitative metrics such as: MRR, precision and recall across different retrieval techniques, RAG variants (Naive, Advanced, Modular), and LLMs to identify the best performing combination.  
-	To demonstrate how a Yoruba-specific RAG system, using the best performing RAG variant and large language model, enhances access to linguistic knowledge and promotes Yoruba language and culture preservation through human evaluations of accuracy, relevance, and cultural fidelity.


---

## Data Sources
The system integrates multiple Yoruba language resources, including:

- **BBC Yorùbá News Corpus** (scraped for academic research purposes)
- **Yankari Yoruba Dataset**
- **Niger-Volta LTI Yoruba Text Corpus**
- **Digitised Cultural Texts**
  - *Àsà àti Ìṣẹ̀ Yorùbá* – G. B. A. Odùnjọ́
  - *Àwọn Àṣà àti Òrìṣà Ilé Yorùbá* – L. J. B. Eades

All datasets are used strictly for non-commercial, academic research and cultural preservation purposes. Here is the link of the dataset : https://drive.google.com/drive/folders/1h7iKCPfQYCbR-cir2TJZE9XoMAlpSLtV?usp=sharing

---

## Yoruba Linguistic Pre-processing
Language-specific preprocessing techniques implemented in this project include:

- Orthographic normalisation and diacritics handling
- Tone-aware tokenisation
- Morphological and word-structure analysis
- Identification and preservation of idiomatic and culturally embedded expressions
- Text cleaning and segmentation for retrieval and generation tasks

---

## Retrieval Techniques
The retrieval component evaluates and compares the following approaches:

- **Sparse Retrieval**  
  - BM25-based lexical retrieval

- **Dense Retrieval**  
  - Transformer-based embeddings using AfriBERTa and related models

- **Hybrid Retrieval**  
  - Combination of sparse and dense retrieval strategies on both (Reciprocal Rank Fusion (RRF) or Relative Score Fusion)

Retrieval performance is analysed across multiple thematic domains relevant to Yoruba sociocultural contexts.

---

## Retrieval-Augmented Generation (RAG) Models
The system implements and evaluates multiple RAG variants, including:

- **Naïve RAG**  
  - Direct retrieval and generation without advanced filtering

- **Advanced RAG**  
  - Context filtering, reranking, and improved prompt construction

- **Modular RAG**  
  - Domain-aware retrievers and modular generation pipelines

Each variant is evaluated for fluency, relevance, and cultural accuracy.

---

## System Architecture and Deployment
The repository includes modules for:

- Backend API design
- Vector database integration
- Modular inference pipelines
- Scalable deployment configurations
- User interaction interfaces

The system is designed to support extensibility and reproducibility.

---

## Evaluation Methodology
Evaluation is conducted using a mixed-method approach:

- **Quantitative Evaluation**
  - Retrieval metrics (e.g., relevance and ranking effectiveness)

- **Human Evaluation**
  - Questionnaire-based assessment
  - Evaluation criteria include:
    - Fluency
    - Relevance
    - Cultural and contextual accuracy

---

### Citation
- Akpobi, M., 2024. Yankari: A Monolingual Yoruba Dataset. https://doi.org/10.48550/arXiv.2412.03334.
- Orife, I., Fasubaa, T. and Wahab, O., 2018. yoruba-text. Available at: <https://github.com/Niger-Volta-LTI/yoruba-text> [Accessed 8 January 2026].
- Orife, I., Kreutzer, J., Sibanda, B., Whitenack, D., Siminyu, K., Martinus, L., Ali, J.T., Abbott, J., Marivate, V., Kabongo, S., Meressa, M., Murhabazi, E., Ahia, O., Biljon, E. van, Ramkilowan, A., Akinfaderin, A., Öktem, A., Akin, W., Kioko, G., Degila, K., Kamper, H., Dossou, B., Emezue, C., Ogueji, K. and Bashir, A., 2020. Masakhane -- Machine Translation For Africa. https://doi.org/10.48550/arXiv.2003.11529.

