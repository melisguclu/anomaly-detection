import ModelInformationLayout from '../components/model-information-layout';

const PaDIM = () => {
  const benefitsData = [
    {
      title: "Pre-trained Features",
      description: "Leverages strong representations from networks like ResNet without needing fine-tuning."
    },
    {
      title: "Local Patch Modeling",
      description: "Models the distribution of features at each patch location, enabling fine-grained anomaly localization."
    },
    {
      title: "Statistical Rigor",
      description: "Uses well-understood probabilistic distance metrics (Mahalanobis) to detect deviations."
    },
    {
      title: "Label-Free Training",
      description: "Requires only normal data for training — no need for defect labels."
    }
  ];

  const keyComponents = [
    {
      title: "Pre-trained CNN Backbone",
      description: "Features are extracted from multiple layers of a CNN (e.g., ResNet-18), allowing both low-level and high-level characteristics to be captured."
    },
    {
      title: "Per-Patch Embedding Distribution",
      description: "For each spatial location (i,j), PaDiM computes the mean vector (μᵢⱼ) and covariance matrix (Σᵢⱼ) across all normal training embeddings."
    },
    {
      title: "Anomaly Score via Mahalanobis Distance",
      description: "During testing, the Mahalanobis distance between the test patch and the learned Gaussian parameters (μᵢⱼ, Σᵢⱼ) is used to generate an anomaly map."
    },
    {
      title: "Anomaly Heatmap",
      description: "The resulting pixel-wise scores are visualized as a heatmap highlighting potential defects."
    }
  ];

  return (
    <ModelInformationLayout 
      modelName="PaDiM"
      modelSubtitle="Patch Distribution Modeling for Advanced Anomaly Detection"
      modelDescription="PaDiM (Patch Distribution Modeling) is a powerful unsupervised anomaly detection method that models the distribution of normal features extracted from a pre-trained convolutional neural network (CNN). It is particularly effective in detecting subtle anomalies in industrial inspection tasks, such as texture or structural defects."
      modelArchitectureImage="/padim.png"
      howItWorksText="PaDiM computes multivariate Gaussian distributions for each spatial location in the feature maps of a pre-trained CNN using only normal (defect-free) images. These Gaussian models represent the normal patterns for each patch in the image. At inference time, PaDiM compares the extracted features of a test image with the corresponding learned Gaussian distribution using Mahalanobis distance. If the distance is high, it indicates that the patch is likely anomalous."
      keyComponents={keyComponents}
      benefits={benefitsData}
    />
  );
};

export default PaDIM; 