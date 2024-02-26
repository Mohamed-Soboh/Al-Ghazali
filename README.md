# EL-Gazali
The software offers a robust learning platform tailored for researchers aiming to assess authorship attribution in Al-Ghazali's manuscripts. It empowers users with fine-grained control over algorithmic parameters and methodologies employed in the analysis.

BERT (Bidirectional Encoder Representations from Transformers): This algorithm is employed for producing word embeddings. BERT has limitations on the input length it can process, which necessitates breaking the text into segments.

Convolutional Neural Networks (CNNs): These networks are utilized for classification purposes. The CNNs are trained iteratively, with each iteration potentially involving the training of a new model, subject to validation accuracy thresholds.

K-Means Clustering: This clustering algorithm is used to partition the data into clusters based on the embeddings produced by BERT and the classifications made by the CNNs.

