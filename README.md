# Plant-Disease-Detection
This project focuses on evaluating deep learning architectures—MobileNetV2, ResNet50, and EfficientNetB0—for plant disease classification using the PlantVillage dataset. The dataset was preprocessed through class balancing, augmentation, and image standardization to improve reliability. Each model was trained and evaluated using key metrics such as accuracy, precision, recall, and F1-score.

The study highlights the trade-off between model efficiency and accuracy:

EfficientNetB0 achieved the highest accuracy (≈98%), making it the most suitable for applications requiring maximum precision.

MobileNetV2, while less accurate, is the fastest and most lightweight, ideal for real-time mobile deployment.

ResNet50 provided balanced performance but lagged behind EfficientNet in accuracy.

Results showed excellent overall performance across most of the 21 plant and disease classes, though some diseases (e.g., Corn Cercospora Leaf Spot) remained challenging due to similarity with other conditions or limited training samples.

The project demonstrates that improving dataset quality is just as crucial as model selection. It concludes that model choice should depend on deployment needs—EfficientNetB0 for maximum accuracy and MobileNetV2 for faster real-time applications. A simple browser-based interface was also proposed, allowing users to upload leaf images for instant disease predictions.
