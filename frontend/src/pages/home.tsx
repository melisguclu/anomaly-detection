import { Box, Container, Typography, Grid } from '@mui/material';
import ChatbotAnimation from '../components/chatbot-animation';

const Home = () => {
  return (
    <Box >
      {/* Chatbot Animation Section */}
      {/* Hero Section */}
      <Box sx={{ py: { xs: 8, md: 12 } }}>
        <Container maxWidth="lg">
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={6}>
              <Typography 
                variant="h1" 
                component="h1" 
                gutterBottom
                sx={{
                  fontSize: { xs: '2.5rem', md: '3.5rem' },
                  fontWeight: 700,
                  lineHeight: 1.2,
                  mb: 3,
                  color: '#1a237e',
                }}
              >
                Wood Surface Anomaly Detection
              </Typography>
              <Typography 
                variant="h5" 
                sx={{ 
                  mb: 4,
                  color: '#666',
                  lineHeight: 1.6,
                }}
              >
                Advanced unsupervised learning techniques for detecting defects and anomalies in wood surfaces using PaDIM and STFPM models
              </Typography>

      <ChatbotAnimation />

        
            </Grid>

          </Grid>
        </Container>
      </Box>

      {/* Description Section */}
      <Container maxWidth="lg" sx={{ mb: 8 }}>
        <Grid container spacing={4}>
          <Grid item xs={12} md={6}>
            <Typography 
              variant="h4" 
              component="h2" 
              gutterBottom
              sx={{ 
                fontWeight: 700,
                color: '#1a237e',
                mb: 3,
              }}
            >
              About Our Solution
            </Typography>
            <Typography 
              variant="body1" 
              sx={{ 
                color: '#666',
                lineHeight: 1.8,
                mb: 3,
              }}
            >
              Our platform specializes in detecting anomalies on wood surfaces using state-of-the-art unsupervised learning techniques. 
              We employ two powerful models:
            </Typography>
            <Typography 
              component="ul" 
              sx={{ 
                pl: 2, 
                mb: 3,
                '& li': {
                  color: '#666',
                  lineHeight: 1.8,
                  mb: 1,
                }
              }}
            >
              <Typography component="li">
                <strong>PaDIM (Patch Distribution Modeling):</strong> Analyzes the distribution of image patches to identify 
                anomalies in wood texture and surface patterns.
              </Typography>
              <Typography component="li">
                <strong>STFPM (Student-Teacher Feature Pyramid Matching):</strong> Uses a student-teacher network architecture 
                to detect defects by comparing feature pyramid representations.
              </Typography>
            </Typography>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography 
              variant="h4" 
              component="h2" 
              gutterBottom
              sx={{ 
                fontWeight: 700,
                color: '#1a237e',
                mb: 3,
              }}
            >
              Key Features
            </Typography>
            <Typography 
              variant="body1" 
              sx={{ 
                color: '#666',
                lineHeight: 1.8,
                mb: 3,
              }}
            >
              Our unsupervised approach offers several advantages for wood surface inspection:
            </Typography>
            <Typography 
              component="ul" 
              sx={{ 
                pl: 2, 
                mb: 3,
                '& li': {
                  color: '#666',
                  lineHeight: 1.8,
                  mb: 1,
                }
              }}
            >
              <Typography component="li">
                No need for labeled defect data - learns from normal wood surface patterns
              </Typography>
              <Typography component="li">
                Detects various types of defects including knots, cracks, and surface irregularities
              </Typography>
              <Typography component="li">
                Real-time analysis with high accuracy and low false positive rates
              </Typography>
              <Typography component="li">
                Works with different wood types and surface finishes
              </Typography>
            </Typography>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default Home; 