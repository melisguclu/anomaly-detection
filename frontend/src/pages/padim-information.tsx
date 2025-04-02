
import { Container, Typography, Box, Paper, Grid } from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';

const PaDIM = () => {
  return (
    <Box sx={{ 
      minHeight: '100vh', 
      bgcolor: '#f5f5f5',
      pt: 8,
      pb: 12,
    }}>
      <Container maxWidth="lg">
        <Box sx={{ textAlign: 'center', mb: 8 }}>
          <Typography 
            variant="h2" 
            component="h1" 
            gutterBottom
            sx={{ 
              fontWeight: 700,
              color: '#1a237e',
            }}
          >
            PaDIM Model
          </Typography>
          <Typography 
            variant="h6" 
            color="text.secondary"
            sx={{ maxWidth: '800px', mx: 'auto', mb: 4 }}
          >
            Patch Distribution Modeling for Industrial Anomaly Detection
          </Typography>
        </Box>

        <Grid container spacing={4}>
          <Grid item xs={12} md={6}>
            <Paper
              elevation={2}
              sx={{
                p: 4,
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                backgroundColor: 'white',
              }}
            >
              <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
                Overview
              </Typography>
              <Typography color="text.secondary" sx={{ lineHeight: 1.8, mb: 3 }}>
                PaDIM is a novel approach for industrial anomaly detection that leverages patch distribution modeling. 
                It's particularly effective for detecting defects in industrial images by analyzing the distribution of image patches.
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <InfoIcon sx={{ mr: 1, color: '#1a237e' }} />
                <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
                  Key Features
                </Typography>
              </Box>
              <Typography component="ul" sx={{ pl: 2, mb: 3 }}>
                <Typography component="li" sx={{ mb: 1 }}>
                  Patch-based analysis for detailed defect detection
                </Typography>
                <Typography component="li" sx={{ mb: 1 }}>
                  Distribution modeling for robust anomaly detection
                </Typography>
                <Typography component="li" sx={{ mb: 1 }}>
                  Efficient processing of industrial images
                </Typography>
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={12} md={6}>
            <Paper
              elevation={2}
              sx={{
                p: 4,
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                backgroundColor: 'white',
              }}
            >
              <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
                Technical Details
              </Typography>
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 500, mb: 1 }}>
                  Architecture
                </Typography>
                <Typography color="text.secondary" sx={{ lineHeight: 1.8 }}>
                  The model uses a pre-trained backbone network to extract features from image patches, 
                  followed by a distribution modeling module that learns the normal pattern distribution.
                </Typography>
              </Box>
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 500, mb: 1 }}>
                  Performance
                </Typography>
                <Typography color="text.secondary" sx={{ lineHeight: 1.8 }}>
                  PaDIM achieves state-of-the-art performance on various industrial datasets, 
                  with high accuracy in defect detection and low false positive rates.
                </Typography>
              </Box>
              <Box>
                <Typography variant="subtitle1" sx={{ fontWeight: 500, mb: 1 }}>
                  Applications
                </Typography>
                <Typography color="text.secondary" sx={{ lineHeight: 1.8 }}>
                  Ideal for quality control in manufacturing, surface defect detection, 
                  and industrial inspection tasks.
                </Typography>
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default PaDIM; 