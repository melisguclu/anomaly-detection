import { useState, useCallback, useEffect } from 'react';
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
  LinearProgress,
  Alert,
  Fade,
  Zoom,
  Grow,
  keyframes,
  Modal,
  IconButton,
  Tooltip,
  Divider,
  Grid
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import SendIcon from '@mui/icons-material/Send';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import CompareIcon from '@mui/icons-material/Compare';
import CloseIcon from '@mui/icons-material/Close';
import ModelSelector, { ModelType } from '../components/model-selector';

// Custom keyframes for animations
const pulse = keyframes`
  0% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.05);
    opacity: 0.8;
  }
  100% {
    transform: scale(1);
    opacity: 1;
  }
`;

// Processing steps
const steps = [
  {
    label: 'Preprocessing',
    subSteps: [
      { label: 'Image Normalization', description: 'Rescales pixel values to the range [0,1].' },
    ]
  },
  {
    label: 'Unsupervised Model Inference',
    subSteps: [
      { label: 'Anomaly Segmentation', description: 'Generates a mask highlighting potential defects.' }
    ]
  },
  {
    label: 'Postprocessing & Evaluation',
    subSteps: [
      { label: 'Thresholding', description: 'Applies a threshold to the anomaly score to classify image as normal or defective.' },
      { label: 'Comparison with Ground Truth', description: 'Compares the predicted mask with the ground-truth mask.' },
      
    ]
  }
];

// Calculate total steps for progress tracking
const totalSteps = steps.reduce((acc, step) => acc + step.subSteps.length, 0);

// Modal style for full-screen image viewing
const modalStyle = {
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: '90%',
  maxWidth: '1200px',
  bgcolor: 'background.paper',
  boxShadow: 24,
  p: 4,
  borderRadius: 2,
  maxHeight: '90vh',
  overflow: 'auto',
};

const Detect = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeStep, setActiveStep] = useState(-1);
  const [activeSubStep, setActiveSubStep] = useState(-1);
  const [result, setResult] = useState<{ score: number; result_image: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [progressValue, setProgressValue] = useState(0);
  const [currentStepProgress, setCurrentStepProgress] = useState(0);
  const [openModal, setOpenModal] = useState(false);
  const [openComparisonModal, setOpenComparisonModal] = useState(false);
  const [selectedModel, setSelectedModel] = useState<ModelType>('padim');
  const [showUploadSection, setShowUploadSection] = useState(true);

  const resetDetection = () => {
    setSelectedImage(null);
    setImagePreview('');
    setResult(null);
    setError(null);
    setShowResults(false);
    setShowUploadSection(true);
  };

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    setSelectedImage(file);
    const preview = URL.createObjectURL(file);
    setImagePreview(preview);
    setResult(null);
    setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png'] },
    multiple: false
  });

  // Calculate overall progress based on completed steps
  useEffect(() => {
    if (isProcessing) {
      // Calculate progress based on completed steps
      let completedSteps = 0;
      
      // Count completed steps from previous steps
      for (let i = 0; i < activeStep; i++) {
        completedSteps += steps[i].subSteps.length;
      }
      
      // Add progress from current step
      if (activeStep >= 0) {
        completedSteps += activeSubStep;
      }
      
      // Calculate percentage - ensure it reaches 100% when all steps are completed
      const percentage = Math.min(Math.round((completedSteps / totalSteps) * 100), 100);
      setProgressValue(percentage);
    } else {
      setProgressValue(0);
    }
  }, [activeStep, activeSubStep, isProcessing]);

  const processImage = async () => {
    if (!selectedImage) return;

    setIsProcessing(true);
    setActiveStep(0);
    setActiveSubStep(0);
    setError(null);
    setShowResults(false);
    setCurrentStepProgress(0);
    setShowUploadSection(false);

    try {
      const formData = new FormData();
      formData.append('file', selectedImage);

      // Determine the endpoint based on the selected model
      let endpoint = '';
      if (selectedModel === 'padim') {
        endpoint = 'http://localhost:8000/api/v1/anomaly/detect';
      } else if (selectedModel === 'stfpm') {
        endpoint = 'http://localhost:8000/api/v1/stfpm/detect';
      } else if (selectedModel === 'efficientad') {
        endpoint = 'http://localhost:8000/api/v1/efficientad/detect';
      }
      

      console.log(`Sending request to backend using ${selectedModel} model...`);
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      console.log('Response status:', response.status);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        console.error('Error response:', errorData);
        throw new Error(errorData?.detail || 'Failed to process image');
      }

      const data = await response.json();
      console.log('Received data:', data);
      setResult(data);

      // Process Preprocessing step with enhanced animations
      setActiveStep(0);
      for (let i = 0; i < steps[0].subSteps.length; i++) {
        setActiveSubStep(i);
        // Simulate progress within each substep
        for (let j = 0; j <= 100; j += 10) {
          setCurrentStepProgress(j);
          await new Promise(resolve => setTimeout(resolve, 150));
        }
        await new Promise(resolve => setTimeout(resolve, 500));
      }

      // Process Model Inference step with enhanced animations
      setActiveStep(1);
      setActiveSubStep(0);
      for (let i = 0; i < steps[1].subSteps.length; i++) {
        setActiveSubStep(i);
        // Simulate progress within each substep
        for (let j = 0; j <= 100; j += 10) {
          setCurrentStepProgress(j);
          await new Promise(resolve => setTimeout(resolve, 200));
        }
        await new Promise(resolve => setTimeout(resolve, 500));
      }

      // Process Postprocessing step with enhanced animations
      setActiveStep(2);
      setActiveSubStep(0);
      for (let i = 0; i < steps[2].subSteps.length; i++) {
        setActiveSubStep(i);
        // Simulate progress within each substep
        for (let j = 0; j <= 100; j += 10) {
          setCurrentStepProgress(j);
          await new Promise(resolve => setTimeout(resolve, 120));
        }
        await new Promise(resolve => setTimeout(resolve, 500));
      }
      
      // Ensure progress reaches 100% before showing results
      setProgressValue(100);
      
      // Add a final delay before showing results
      await new Promise(resolve => setTimeout(resolve, 1000));
      setShowResults(true);
    } catch (err) {
      console.error('Error details:', err);
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsProcessing(false);
      setActiveStep(-1);
      setActiveSubStep(-1);
      setCurrentStepProgress(0);
    }
  };

  const handleOpenModal = () => setOpenModal(true);
  const handleCloseModal = () => setOpenModal(false);
  const handleOpenComparisonModal = () => setOpenComparisonModal(true);
  const handleCloseComparisonModal = () => setOpenComparisonModal(false);

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Box sx={{ pt: 4, pb: 6 }}>
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
                textShadow: '2px 2px 4px rgba(0,0,0,0.1)',
              }}
            >
              Detect Anomalies
            </Typography>
          </Box>

          {error && (
            <Fade in={!!error}>
              <Alert severity="error" sx={{ mb: 4 }}>
                {error}
              </Alert>
            </Fade>
          )}

          <ModelSelector 
            selectedModel={selectedModel} 
            onModelChange={setSelectedModel}
            disabled={isProcessing}
          />

          {showUploadSection && (
            <Paper
              elevation={0}
              sx={{
                p: 4,
                mb: 4,
                border: '2px dashed',
                borderColor: isDragActive ? '#1a237e' : '#e0e0e0',
                borderRadius: 2,
                bgcolor: isDragActive ? 'rgba(26, 35, 126, 0.04)' : 'white',
                position: 'relative',
                overflow: 'hidden',
              }}
            >
              <Box
                {...getRootProps()}
                sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  cursor: 'pointer',
                  position: 'relative',
                  zIndex: 1,
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
          )}

          {/* Preview and Process Section */}
          {imagePreview && (
            <Zoom in={!!imagePreview} timeout={500}>
              <Box sx={{ mb: 4 }}>
                <Paper
                  elevation={0}
                  sx={{
                    p: 3,
                    border: '1px solid #e0e0e0',
                    borderRadius: 2,
                    position: 'relative',
                    overflow: 'hidden',
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
                          boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
                          transition: 'all 0.3s ease',
                          '&:hover': {
                            transform: 'scale(1.02)',
                            boxShadow: '0 8px 16px rgba(0,0,0,0.15)',
                          },
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
                    {isProcessing && (
                      <Grow in={isProcessing} timeout={800}>
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
                                      transition: 'all 0.3s ease',
                                      transform: activeStep === index ? 'scale(1.05)' : 'scale(1)',
                                    }}
                                  >
                                    {step.label}
                                  </Typography>
                                </StepLabel>
                                <StepContent>
                                  {step.subSteps.map((subStep, subIndex) => (
                                    <Box 
                                      key={subStep.label} 
                                      sx={{ 
                                        mb: 2,
                                        transition: 'all 0.3s ease',
                                        transform: activeStep === index && activeSubStep === subIndex ? 'translateX(10px)' : 'translateX(0)',
                                        opacity: activeStep === index && activeSubStep === subIndex ? 1 : 0.7,
                                      }}
                                    >
                                      <Typography
                                        variant="body2"
                                        sx={{
                                          color: activeStep === index && activeSubStep === subIndex
                                            ? '#1a237e'
                                            : 'text.secondary',
                                          fontWeight: activeStep === index && activeSubStep === subIndex
                                            ? 600
                                            : 400,
                                          display: 'flex',
                                          alignItems: 'center',
                                        }}
                                      >
                                        {subStep.label}
                                        {activeStep === index && activeSubStep === subIndex && (
                                          <CircularProgress
                                            size={16}
                                            sx={{ 
                                              ml: 1, 
                                              color: '#1a237e',
                                              animation: `${pulse} 1.5s infinite ease-in-out`,
                                            }}
                                          />
                                        )}
                                      </Typography>
                                      <Typography
                                        variant="body2"
                                        color="text.secondary"
                                        sx={{ 
                                          fontSize: '0.875rem',
                                          transition: 'all 0.3s ease',
                                          opacity: activeStep === index && activeSubStep === subIndex ? 1 : 0.7,
                                        }}
                                      >
                                        {subStep.description}
                                      </Typography>
                                      {activeStep === index && activeSubStep === subIndex && (
                                        <LinearProgress
                                          variant="determinate"
                                          value={currentStepProgress}
                                          sx={{
                                            mt: 1,
                                            height: 8,
                                            borderRadius: 4,
                                            bgcolor: 'rgba(26, 35, 126, 0.1)',
                                          }}
                                        />
                                      )}
                                    </Box>
                                  ))}
                                </StepContent>
                              </Step>
                            ))}
                          </Stepper>
                          
                          {/* Overall progress indicator */}
                          <Box sx={{ mt: 3, position: 'relative' }}>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Overall Progress
                            </Typography>
                            <LinearProgress 
                              variant="determinate" 
                              value={progressValue} 
                              sx={{ 
                                height: 10, 
                                borderRadius: 5,
                                bgcolor: 'rgba(26, 35, 126, 0.1)',
                              }} 
                            />
                            <Typography 
                              variant="caption" 
                              sx={{ 
                                position: 'absolute', 
                                right: 0, 
                                top: 0,
                                color: '#1a237e',
                                fontWeight: 600,
                              }}
                            >
                              {progressValue}%
                            </Typography>
                          </Box>
                        </Box>
                      </Grow>
                    )}

                    {/* Results Section */}
                    {showResults && result && (
                      <Grow in={showResults} timeout={1000}>
                        <Box sx={{ flex: 1 }}>
                          <Typography variant="h6" gutterBottom>
                            Detection Results
                          </Typography>
                          
                          <Paper 
                            elevation={0} 
                            sx={{ 
                              p: 2, 
                              mb: 2,
                              bgcolor: 'rgba(26, 35, 126, 0.05)',
                              borderRadius: 2,
                              transition: 'all 0.3s ease',
                              '&:hover': {
                                bgcolor: 'rgba(26, 35, 126, 0.1)',
                              },
                            }}
                          >
                            <Typography variant="body1" gutterBottom>
                              Anomaly Score: <strong>{result.score.toFixed(4)}</strong>
                            </Typography>
                            
                          </Paper>
                          
                          <Box sx={{ position: 'relative', mb: 2 }}>
                            <Box
                              component="img"
                              src={result?.result_image ? `http://localhost:8000${result.result_image}` : ''}
                              alt="Detection results"
                              sx={{
                                width: '100%',
                                height: 'auto',
                                borderRadius: 1,
                                boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
                                transition: 'all 0.3s ease',
                                '&:hover': {
                                  transform: 'scale(1.02)',
                                  boxShadow: '0 8px 16px rgba(0,0,0,0.15)',
                                },
                              }}
                            />
                            <Box 
                              sx={{ 
                                position: 'absolute', 
                                top: 10, 
                                right: 10, 
                                display: 'flex', 
                                gap: 1,
                                bgcolor: 'rgba(255, 255, 255, 0.8)',
                                borderRadius: 1,
                                p: 0.5,
                                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                              }}
                            >
                              <Tooltip title="View fullscreen">
                                <IconButton 
                                  size="small" 
                                  onClick={handleOpenModal}
                                  sx={{ color: '#1a237e' }}
                                >
                                  <FullscreenIcon />
                                </IconButton>
                              </Tooltip>
                              <Tooltip title="Compare with original">
                                <IconButton 
                                  size="small" 
                                  onClick={handleOpenComparisonModal}
                                  sx={{ color: '#1a237e' }}
                                >
                                  <CompareIcon />
                                </IconButton>
                              </Tooltip>
                            </Box>
                          </Box>
                        </Box>
                      </Grow>
                    )}
                  </Box>
                </Paper>
              </Box>
            </Zoom>
          )}
          
          {/* Fullscreen Modal */}
          <Modal
            open={openModal}
            onClose={handleCloseModal}
            aria-labelledby="result-image-modal"
            aria-describedby="fullscreen-result-image"
          >
            <Box sx={modalStyle}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" component="h2">
                  Detection Result
                </Typography>
                <IconButton onClick={handleCloseModal} size="small">
                  <CloseIcon />
                </IconButton>
              </Box>
              <Divider sx={{ mb: 2 }} />
              <Box
                component="img"
                src={result?.result_image ? `http://localhost:8000${result.result_image}` : ''}
                alt="Detection results"
                sx={{
                  width: '100%',
                  height: 'auto',
                  maxHeight: '70vh',
                  objectFit: 'contain',
                }}
              />
              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body1">
                  Anomaly Score: <strong>{result?.score.toFixed(4)}</strong>
                </Typography>
                <Button 
                  variant="outlined" 
                  size="small" 
                  onClick={handleCloseModal}
                  startIcon={<CloseIcon />}
                >
                  Close
                </Button>
              </Box>
            </Box>
          </Modal>
          
          {/* Comparison Modal */}
          <Modal
            open={openComparisonModal}
            onClose={handleCloseComparisonModal}
            aria-labelledby="comparison-modal"
            aria-describedby="image-comparison"
          >
            <Box sx={modalStyle}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" component="h2">
                  Image Comparison
                </Typography>
                <IconButton onClick={handleCloseComparisonModal} size="small">
                  <CloseIcon />
                </IconButton>
              </Box>
              <Divider sx={{ mb: 2 }} />
              <Grid container spacing={2}>
                <Grid component="div">
                  <Paper 
                    elevation={0} 
                    sx={{ 
                      p: 2, 
                      border: '1px solid #e0e0e0',
                      borderRadius: 2,
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column',
                    }}
                  >
                    <Typography variant="subtitle1" gutterBottom>
                      Original Image
                    </Typography>
                    <Box
                      component="img"
                      src={imagePreview}
                      alt="Original image"
                      sx={{
                        width: '100%',
                        height: 'auto',
                        maxHeight: '60vh',
                        objectFit: 'contain',
                        borderRadius: 1,
                        flexGrow: 1,
                      }}
                    />
                  </Paper>
                </Grid>
                <Grid component="div">
                  <Paper 
                    elevation={0} 
                    sx={{ 
                      p: 2, 
                      border: '1px solid #e0e0e0',
                      borderRadius: 2,
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column',
                    }}
                  >
                    <Typography variant="subtitle1" gutterBottom>
                      Detection Result
                    </Typography>
                    <Box
                      component="img"
                      src={result?.result_image ? `http://localhost:8000${result.result_image}` : ''}
                      alt="Detection results"
                      sx={{
                        width: '100%',
                        height: 'auto',
                        maxHeight: '60vh',
                        objectFit: 'contain',
                        borderRadius: 1,
                        flexGrow: 1,
                      }}
                    />
                  </Paper>
                </Grid>
              </Grid>
              <Box sx={{ mt: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body1">
                  Anomaly Score: <strong>{result?.score.toFixed(4)}</strong>
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {result?.score && result.score > 0.6
                    ? '⚠️ Likely Defective'
                    : '✅ Likely Normal'}
                </Typography>

                <Button 
                  variant="outlined" 
                  size="small" 
                  onClick={handleCloseComparisonModal}
                  startIcon={<CloseIcon />}
                >
                  Close
                </Button>
              </Box>
            </Box>
          </Modal>

          {/* Results Section */}
          {showResults && result && (
            <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
              <Button
                variant="contained"
                onClick={resetDetection}
                sx={{
                  backgroundColor: '#1a237e',
                  '&:hover': {
                    backgroundColor: '#0d47a1',
                  },
                }}
              >
                Try Another Image
              </Button>
            </Box>
          )}
        </Container>
      </Box>
    </Container>
  );
};

export default Detect; 