import { useState, useCallback } from 'react';
import { 
  Box, 
  Container, 
  Typography, 
  Button, 
  Paper,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  CircularProgress,
  LinearProgress
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import SendIcon from '@mui/icons-material/Send';

// Processing steps
const steps = [
  {
    label: '🔍 Preprocessing',
    subSteps: [
      { label: 'Background Removal', description: 'Removes the background to focus on the wood surface only.' },
      { label: 'Image Normalization', description: 'Rescales pixel values to the range [0,1].' },
      { label: 'Image Resizing', description: 'Resizes all images to 256x256 pixels.' }
    ]
  },
  {
    label: '🧠 Unsupervised Model Inference',
    subSteps: [
      { label: 'Feature Extraction', description: 'Extracts deep features from the input image.' },
      { label: 'Anomaly Scoring', description: 'Calculates an anomaly score for the input image.' },
      { label: 'Anomaly Segmentation', description: 'Generates a mask highlighting potential defects.' }
    ]
  },
  {
    label: '✅ Postprocessing & Evaluation',
    subSteps: [
      { label: 'Thresholding', description: 'Applies a threshold to the anomaly score to classify image as normal or defective.' },
      { label: 'Comparison with Ground Truth', description: 'Compares the predicted mask with the ground-truth mask.' },
      { label: 'Metric Calculation', description: 'Calculates performance metrics such as F1 Score and IoU.' }
    ]
  }
];

const Detect = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeStep, setActiveStep] = useState(-1);
  const [activeSubStep, setActiveSubStep] = useState(-1);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    setSelectedImage(file);
    const preview = URL.createObjectURL(file);
    setImagePreview(preview);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png'] },
    multiple: false
  });

  const processImage = async () => {
    setIsProcessing(true);
    setActiveStep(0);
    setActiveSubStep(0);

    // Mock processing steps
    for (let i = 0; i < steps.length; i++) {
      setActiveStep(i);
      for (let j = 0; j < steps[i].subSteps.length; j++) {
        setActiveSubStep(j);
        // Simulate processing time
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }

    setIsProcessing(false);
    setActiveStep(-1);
    setActiveSubStep(-1);
  };

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'white', pt: 4, pb: 6 }}>
      <Container maxWidth="lg">
        <Box sx={{ mb: 6, mt: 4 }}>
          <Typography 
            variant="h3" 
            component="h1" 
            align="center"
            gutterBottom
            sx={{ 
              fontWeight: 700,
              color: '#1a237e',
            }}
          >
            Detect Anomalies
          </Typography>
        </Box>

        {/* Upload Section */}
        <Paper
          elevation={0}
          sx={{
            p: 4,
            mb: 4,
            border: '2px dashed',
            borderColor: isDragActive ? '#1a237e' : '#e0e0e0',
            borderRadius: 2,
            bgcolor: isDragActive ? 'rgba(26, 35, 126, 0.04)' : 'white',
            transition: 'all 0.3s ease',
          }}
        >
          <Box
            {...getRootProps()}
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              cursor: 'pointer',
            }}
          >
            <input {...getInputProps()} />
            <CloudUploadIcon sx={{ fontSize: 48, color: '#1a237e', mb: 2 }} />
            <Typography variant="h6" align="center" gutterBottom>
              {isDragActive
                ? 'Drop the wood surface image here'
                : 'Drag and drop a wood surface image here, or click to select'}
            </Typography>
            <Typography variant="body2" color="text.secondary" align="center">
              Supported formats: JPEG, JPG, PNG
            </Typography>
          </Box>
        </Paper>

        {/* Preview and Process Section */}
        {imagePreview && (
          <Box sx={{ mb: 4 }}>
            <Paper
              elevation={0}
              sx={{
                p: 3,
                border: '1px solid #e0e0e0',
                borderRadius: 2,
              }}
            >
              <Box
                sx={{
                  display: 'flex',
                  gap: 4,
                  flexDirection: { xs: 'column', md: 'row' },
                }}
              >
                <Box
                  sx={{
                    flex: 1,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                  }}
                >
                  <Typography variant="h6" gutterBottom>
                    Selected Image
                  </Typography>
                  <Box
                    component="img"
                    src={imagePreview}
                    alt="Selected wood surface"
                    sx={{
                      width: '100%',
                      maxWidth: '400px',
                      height: 'auto',
                      borderRadius: 1,
                      mb: 2,
                    }}
                  />
                  <Button
                    variant="contained"
                    startIcon={<SendIcon />}
                    disabled={isProcessing}
                    onClick={processImage}
                    sx={{
                      backgroundColor: '#1a237e',
                      '&:hover': {
                        backgroundColor: '#0d47a1',
                      },
                    }}
                  >
                    {isProcessing ? 'Processing...' : 'Start Detection'}
                  </Button>
                </Box>

                {/* Processing Steps */}
                <Box sx={{ flex: 1 }}>
                  <Typography variant="h6" gutterBottom>
                    Processing Steps
                  </Typography>
                  <Stepper activeStep={activeStep} orientation="vertical">
                    {steps.map((step, index) => (
                      <Step key={step.label}>
                        <StepLabel>
                          <Typography 
                            sx={{ 
                              color: activeStep === index ? '#1a237e' : 'inherit',
                              fontWeight: activeStep === index ? 700 : 400,
                            }}
                          >
                            {step.label}
                          </Typography>
                        </StepLabel>
                        <StepContent>
                          {step.subSteps.map((subStep, subIndex) => (
                            <Box key={subStep.label} sx={{ mb: 2 }}>
                              <Typography
                                variant="body2"
                                sx={{
                                  color: activeStep === index && activeSubStep === subIndex
                                    ? '#1a237e'
                                    : 'text.secondary',
                                  fontWeight: activeStep === index && activeSubStep === subIndex
                                    ? 600
                                    : 400,
                                }}
                              >
                                {subStep.label}
                                {activeStep === index && activeSubStep === subIndex && (
                                  <CircularProgress
                                    size={16}
                                    sx={{ ml: 1, color: '#1a237e' }}
                                  />
                                )}
                              </Typography>
                              <Typography
                                variant="body2"
                                color="text.secondary"
                                sx={{ fontSize: '0.875rem' }}
                              >
                                {subStep.description}
                              </Typography>
                              {activeStep === index && activeSubStep === subIndex && (
                                <LinearProgress
                                  sx={{
                                    mt: 1,
                                    bgcolor: 'rgba(26, 35, 126, 0.1)',
                                    '& .MuiLinearProgress-bar': {
                                      bgcolor: '#1a237e',
                                    },
                                  }}
                                />
                              )}
                            </Box>
                          ))}
                        </StepContent>
                      </Step>
                    ))}
                  </Stepper>
                </Box>
              </Box>
            </Paper>
          </Box>
        )}
      </Container>
    </Box>
  );
};

export default Detect; 