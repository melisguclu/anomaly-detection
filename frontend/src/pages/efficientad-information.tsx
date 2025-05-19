import ModelInformationLayout from '../components/model-information-layout';

const EfficientAD = () => {
  const benefitsData = [
    {
      title: "Lightweight & Scalable",
      description: "Delivers high performance on low-resource devices, making it suitable for edge deployment in industrial environments."
    },
    {
      title: "Knowledge Distillation",
      description: "Uses a pretrained teacher network to effectively transfer knowledge to the student model trained only on normal data."
    },
    {
      title: "Precise Localization",
      description: "Generates pixel-level anomaly maps to accurately detect and localize defects across a variety of surfaces."
    },
    {
      title: "No Anomaly Labels Needed",
      description: "Trains solely on normal samples, eliminating the need for defect annotations during training."
    }
  ];

  const keyComponents = [
    {
      title: "Teacher-Student Architecture",
      description: "Pretrained teacher guides a student network to learn normal data representations via feature mimicry."
    },
    {
      title: "Asymmetric Feature Comparison",
      description: "Compares student and teacher features across multiple layers to capture both global and local anomalies."
    },
    {
      title: "Dual Anomaly Maps",
      description: "Generates both image-level and pixel-level anomaly maps to support classification and defect localization."
    },
    {
      title: "Efficient Implementation",
      description: "Designed for high-speed inference and minimal resource usage, enabling deployment in real-time systems."
    }
  ];

  return (
    <ModelInformationLayout 
      modelName="EfficientAD"
      modelSubtitle="Efficient Anomaly Detection for Industrial Applications"
      modelDescription="EfficientAD is a state-of-the-art anomaly detection framework designed for industrial surface inspection. It employs an unsupervised learning approach based on a teacher-student architecture. The teacher model is pretrained and fixed, while the student model is trained only on normal samples. By comparing their feature representations, EfficientAD detects deviations that signal potential anomalies. It excels at identifying subtle defects while maintaining high inference speed and low computational cost."
      modelArchitectureImage="/efficientAD.png"
      howItWorksText="EfficientAD works by comparing feature representations between a fixed teacher network and a trainable student network. Both networks share the same architecture. The student is trained to replicate the teacher's outputs for normal images. Discrepancies between the two reveal abnormal patterns. These discrepancies are computed at multiple scales and layers, producing both global (image-level) and local (pixel-level) anomaly maps that are used for classification and segmentation respectively."
      keyComponents={keyComponents}
      benefits={benefitsData}
      footerText="EfficientAD was implemented and evaluated using a wood surface dataset. The model supports unsupervised anomaly detection, providing accurate results using F1 Score and IoU metrics through optimized threshold selection."
    />
  );
};

export default EfficientAD;
