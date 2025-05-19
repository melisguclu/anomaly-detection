import { Container, Typography, Box, Paper, List, ListItem, ListItemIcon, ListItemText, Divider } from '@mui/material';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import PsychologyIcon from '@mui/icons-material/Psychology';
import ArchitectureIcon from '@mui/icons-material/Architecture';
import BenefitsCarousel from './benefits-carousel';

export interface KeyComponent {
  title: string;
  description: string;
}

export interface ModelBenefit {
  title: string;
  description: string;
}

interface ModelInformationLayoutProps {
  modelName: string;
  modelSubtitle: string;
  modelDescription: string;
  modelArchitectureImage: string;
  howItWorksText: string;
  keyComponents: KeyComponent[];
  benefits: ModelBenefit[];
  footerText?: string;
}

const ModelInformationLayout = ({
  modelName,
  modelSubtitle,
  modelDescription,
  modelArchitectureImage,
  howItWorksText,
  keyComponents,
  benefits,
  footerText
}: ModelInformationLayoutProps) => {
  return (
    <Box sx={{ 
      bgcolor: '#f5f5f5',
      position: 'relative',
      '&::before': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: '400px',
        background: 'linear-gradient(135deg, #1a237e 0%, #3949ab 100%)',
        zIndex: 0,
      }
    }}>
      <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1, pt: { xs: 12, md: 16 }, pb: 12 }}>
        {/* Header Section */}
        <Typography 
          variant="h2" 
          component="h1" 
          align="center"
          gutterBottom
          sx={{ 
            fontWeight: 700,
            color: '#ffffff',
            mb: 2,
            textShadow: '2px 2px 4px rgba(0,0,0,0.2)'
          }}
        >
          Understanding {modelName}
        </Typography>
        <Typography
          variant="h5"
          align="center"
          sx={{
            color: '#ffffff',
            mb: 6,
            opacity: 0.9,
            maxWidth: '800px',
            mx: 'auto'
          }}
        >
          {modelSubtitle}
        </Typography>

        {/* Overview Section */}
        <Paper 
          elevation={3}
          sx={{
            p: 4,
            mb: 4,
            borderRadius: 2,
            background: 'linear-gradient(to right bottom, #ffffff, #fafafa)',
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <PsychologyIcon sx={{ fontSize: 40, color: 'primary.main', mr: 2 }} />
            <Typography variant="h4" component="h2" color="primary" sx={{ fontWeight: 600 }}>
              What is {modelName}?
            </Typography>
          </Box>
          <Typography variant="body1" paragraph sx={{ fontSize: '1.1rem', lineHeight: 1.8 }}>
            {modelDescription}
          </Typography>
        </Paper>

        {/* Model Architecture Image */}
        <Paper 
          elevation={3}
          sx={{
            p: 4,
            mb: 4,
            borderRadius: 2,
            background: '#ffffff',
            textAlign: 'center'
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3, justifyContent: 'center' }}>
            <ArchitectureIcon sx={{ fontSize: 40, color: 'primary.main', mr: 2 }} />
            <Typography variant="h4" component="h2" color="primary" sx={{ fontWeight: 600 }}>
              Model Architecture
            </Typography>
          </Box>
          <Box sx={{ 
            position: 'relative',
            '&::after': {
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
            }
          }}>
            <img
              src={modelArchitectureImage}
              alt={`${modelName} Architecture`}
              style={{
                maxWidth: '100%',
                height: 'auto',
                borderRadius: '8px',
                border: '1px solid rgba(0,0,0,0.1)'
              }}
            />
          </Box>
        </Paper>

        {/* How it Works Section */}
        <Paper 
          elevation={3}
          sx={{
            p: 4,
            mb: 4,
            borderRadius: 2,
            background: 'linear-gradient(to right bottom, #ffffff, #fafafa)'
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
            <AutoGraphIcon sx={{ fontSize: 40, color: 'primary.main', mr: 2 }} />
            <Typography variant="h4" component="h2" color="primary" sx={{ fontWeight: 600 }}>
              How It Works
            </Typography>
          </Box>
          <Typography variant="body1" sx={{ fontSize: '1.1rem', lineHeight: 1.8, mb: 4 }}>
            {howItWorksText}
          </Typography>
          <Divider sx={{ my: 4 }} />
          <Typography variant="h5" gutterBottom color="primary" sx={{ fontWeight: 600, mb: 3 }}>
            Key Components
          </Typography>
          <List sx={{ bgcolor: 'background.paper', borderRadius: 2, p: 2 }}>
            {keyComponents.map((component, index) => (
              <ListItem 
                key={index} 
                sx={{ 
                  mb: index < keyComponents.length - 1 ? 2 : 0, 
                  bgcolor: 'rgba(63, 81, 181, 0.05)', 
                  borderRadius: 1 
                }}
              >
                <ListItemIcon>
                  <CheckCircleOutlineIcon color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary={<Typography variant="h6">{component.title}</Typography>}
                  secondary={component.description}
                  secondaryTypographyProps={{ sx: { fontSize: '1rem' } }}
                />
              </ListItem>
            ))}
          </List>
        </Paper>

        {/* Benefits Section */}
        <Paper 
          elevation={3}
          sx={{
            p: 4,
            borderRadius: 2,
            background: 'linear-gradient(to right bottom, #ffffff, #fafafa)',
            overflow: 'hidden'
          }}
        >
          <Typography variant="h4" component="h2" gutterBottom color="primary" sx={{ 
            fontWeight: 600,
            mb: 4,
            textAlign: 'center'
          }}>
            Why Use {modelName}?
          </Typography>
          
          <BenefitsCarousel benefitsData={benefits} />
        </Paper>

        {/* Optional Footer */}
        {footerText && (
          <Typography variant="body2" align="center" sx={{ mt: 6, color: 'text.secondary' }}>
            {footerText}
          </Typography>
        )}
      </Container>
    </Box>
  );
};

export default ModelInformationLayout; 