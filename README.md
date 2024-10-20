# Metadata Extraction

This project focuses on developing a machine learning-based system for metadata extraction from Persian theses and dissertations in DOCX format. The research is divided into two key phases: boundary detection and metadata classification. In the first phase, the system identifies the boundary between header metadata and the main body content of the document. In the second phase, the extracted metadata is classified into predefined categories to enable structured analysis and retrieval. The project leverages various feature extraction techniques and machine learning models to automate the process, improving accuracy and efficiency in handling large collections of academic documents.

## Packages

### `boundary_detection`

This package contains the implementation of the boundary detection phase for metadata and body content. It includes:
- Storage of extracted features.
- Analysis of features to assess their importance.
- Training machine learning models.
- Optimizing model parameters for better performance.

### `Features`

This package implements the features, preprocessing, and variables used in both the boundary detection and metadata classification phases. It covers:
- Definition of all features.
- Methods for calculating each feature.

### `metadata_classification`

This package implements the metadata classification phase. Similar to the boundary detection package, it includes:
- Storage of extracted features.
- Analysis of feature importance.
- Training classification models.
- Optimizing model parameters for better results.








