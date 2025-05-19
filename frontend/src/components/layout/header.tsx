import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box, Container, useScrollTrigger, Slide } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import ScienceIcon from '@mui/icons-material/Science';

interface HideOnScrollProps {
  children: React.ReactElement;
}

function HideOnScroll(props: HideOnScrollProps) {
  const trigger = useScrollTrigger();
  return (
    <Slide appear={false} direction="down" in={!trigger}>
      {props.children}
    </Slide>
  );
}

const Header = () => {
  return (
    <HideOnScroll>
      <AppBar 
        position="fixed" 
        sx={{ 
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(8px)',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        }}
      >
        <Container maxWidth="lg">
          <Toolbar disableGutters>
            <ScienceIcon sx={{ display: { xs: 'none', md: 'flex' }, mr: 1, fontSize: 28 }} />
            <Typography
              variant="h6"
              component={RouterLink}
              to="/"
              sx={{
                flexGrow: 1,
                textDecoration: 'none',
                fontWeight: 'bold',
                letterSpacing: '0.5px',
                color: '#1a237e',
              }}
            >
              Anomaly Detect
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
              <Button
                component={RouterLink}
                to="/padim"
                sx={{
                fontWeight: 'bold',
                  color: '#1a237e',
                }}
              >
                PaDIM
              </Button>
              <Button
                component={RouterLink}
                to="/stfpm"
                sx={{
                fontWeight: 'bold',
                  color: '#1a237e',
                }}
              >
                STFPM
              </Button>
              <Button

                component={RouterLink}
                to="/efficientad"
                sx={{
                  fontWeight: 'bold',
                  color: '#1a237e',
                }}
              >
                EfficientAD
              </Button>
              

              <Button
                variant="contained"
                component={RouterLink}
                to="/detect"
                sx={{
                  backgroundColor: '#2196F3',
                  color: 'white',
                  padding: '8px 24px',
                  '&:hover': {
                    backgroundColor: '#1976D2',
                  },
                }}
              >
                Get Started
              </Button>
            </Box>
          </Toolbar>
        </Container>
      </AppBar>
    </HideOnScroll>
  );
};

export default Header; 