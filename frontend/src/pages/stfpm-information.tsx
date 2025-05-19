import ModelInformationLayout from '../components/model-information-layout';

const STFPM = () => {
  const benefitsData = [
    {
      title: "No Need for Defect Labels",
      description: "STFPM requires only normal (non-defective) images for training."
    },
    {
      title: "High Accuracy",
      description: "The model achieves excellent results in anomaly segmentation, measured by metrics like F1 score and IoU."
    },
    {
      title: "Pixel-Level Localization",
      description: "Unlike classification-only models, STFPM highlights where the anomaly is."
    },
    {
      title: "Easy to Integrate",
      description: "Based on widely used deep learning frameworks (e.g., PyTorch), making it easy to adapt and deploy."
    }
  ];

  const keyComponents = [
    {
      title: "Feature Pyramid Matching",
      description: "Features from multiple levels (layers) of the neural network are extracted and compared. This enables detection of both low-level texture anomalies and high-level structural inconsistencies."
    },
    {
      title: "Anomaly Score",
      description: "The difference between the teacher and student feature embeddings is used to calculate a pixel-wise anomaly score."
    },
    {
      title: "Segmentation Mask",
      description: "A heatmap highlighting anomalous areas is produced, helping localize defects visually."
    }
  ];

  return (
    <ModelInformationLayout 
      modelName="STFPM"
      modelSubtitle="Student-Teacher Feature Pyramid Matching for Advanced Anomaly Detection"
      modelDescription="STFPM (Student-Teacher Feature Pyramid Matching) is an unsupervised anomaly detection and localization model designed for identifying surface defects, particularly effective on visual industrial datasets such as wood textures. It leverages a pre-trained convolutional neural network to learn normal image representations without requiring labeled anomaly data."
      modelArchitectureImage="/STFPM.png"
      howItWorksText="The STFPM framework consists of two networks: a teacher and a student. Both networks share the same architecture (typically ResNet or similar), but only the teacher is pre-trained and frozen. During training, the student network attempts to mimic the feature representations of the teacher using only normal samples. Because the student never sees defective data, it fails to reproduce feature representations accurately when presented with anomalous regions at test time."
      keyComponents={keyComponents}
      benefits={benefitsData}
    />
  );
};

export default STFPM; 