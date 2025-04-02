import { Container, Typography, Box, Paper, Grid } from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';

const STFPM = () => {
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
            STFPM Model
          </Typography>
          <Typography 
            variant="h6" 
            color="text.secondary"
            sx={{ maxWidth: '800px', mx: 'auto', mb: 4 }}
          >
            Student-Teacher Feature Pyramid Matching for Industrial Anomaly Detection
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
                STFPM is an innovative approach that uses a student-teacher network architecture with feature pyramid matching 
                for detecting anomalies in industrial images. It combines the benefits of knowledge distillation with multi-scale feature analysis.
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <InfoIcon sx={{ mr: 1, color: '#1a237e' }} />
                <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
                  Key Features
                </Typography>
              </Box>
              <Typography component="ul" sx={{ pl: 2, mb: 3 }}>
                <Typography component="li" sx={{ mb: 1 }}>
                  Student-teacher network architecture
                </Typography>
                <Typography component="li" sx={{ mb: 1 }}>
                  Multi-scale feature pyramid matching
                </Typography>
                <Typography component="li" sx={{ mb: 1 }}>
                  Efficient anomaly detection
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
                  The model employs a teacher network pre-trained on ImageNet and a student network that learns 
                  to match the teacher's feature pyramid representations.
                </Typography>
              </Box>
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 500, mb: 1 }}>
                  Performance
                </Typography>
                <Typography color="text.secondary" sx={{ lineHeight: 1.8 }}>
                  STFPM demonstrates superior performance in detecting various types of anomalies, 
                  with high accuracy and low computational overhead.
                </Typography>
              </Box>
              <Box>
                <Typography variant="subtitle1" sx={{ fontWeight: 500, mb: 1 }}>
                  Applications
                </Typography>
                <Typography color="text.secondary" sx={{ lineHeight: 1.8 }}>
                  Suitable for industrial quality control, defect detection in manufacturing, 
                  and real-time anomaly monitoring.
                </Typography>
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default STFPM; 