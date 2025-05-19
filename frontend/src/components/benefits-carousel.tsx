import { Box, Paper, Typography, IconButton } from '@mui/material';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import { useState, useEffect } from 'react';

interface Benefit {
  title: string;
  description: string;
}

interface BenefitsCarouselProps {
  benefitsData: Benefit[];
  autoRotateInterval?: number;
}

const BenefitsCarousel = ({ benefitsData, autoRotateInterval = 5000 }: BenefitsCarouselProps) => {
  const [currentCard, setCurrentCard] = useState(0);

  const handleNext = () => {
    setCurrentCard((prev) => (prev + 1) % benefitsData.length);
  };

  const handlePrev = () => {
    setCurrentCard((prev) => (prev - 1 + benefitsData.length) % benefitsData.length);
  };

  useEffect(() => {
    if (autoRotateInterval > 0) {
      const timer = setInterval(() => {
        handleNext();
      }, autoRotateInterval);

      return () => clearInterval(timer);
    }
  }, [autoRotateInterval]);

  return (
    <Box sx={{ 
      position: 'relative',
      height: '250px',
      width: '100%',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center'
    }}>
      <IconButton 
        onClick={handlePrev}
        sx={{ 
          position: 'absolute',
          left: 0,
          zIndex: 2,
          color: 'primary.main',
          '&:hover': { backgroundColor: 'rgba(63, 81, 181, 0.1)' }
        }}
      >
        <ArrowBackIcon />
      </IconButton>

      <Box sx={{ 
        position: 'relative',
        width: '300px',
        height: '200px',
        perspective: '1000px'
      }}>
        {benefitsData.map((benefit, index) => {
          const isActive = index === currentCard;
          const isPrev = (currentCard === 0 ? benefitsData.length - 1 : currentCard - 1) === index;
          const isNext = (currentCard + 1) % benefitsData.length === index;
          
          return (
            <Paper
              key={index}
              elevation={3}
              sx={{
                position: 'absolute',
                width: '100%',
                height: '100%',
                p: 3,
                background: 'linear-gradient(135deg, #3949ab 0%, #1a237e 100%)',
                color: 'white',
                transition: 'all 0.6s ease-in-out',
                transform: isActive 
                  ? 'translateX(0) scale(1) rotateY(0)'
                  : isPrev
                    ? 'translateX(-100%) scale(0.8) rotateY(45deg)'
                    : isNext
                      ? 'translateX(100%) scale(0.8) rotateY(-45deg)'
                      : 'translateX(0) scale(0.6) rotateY(90deg)',
                opacity: isActive ? 1 : isPrev || isNext ? 0.7 : 0,
                zIndex: isActive ? 3 : isPrev || isNext ? 2 : 1,
                pointerEvents: isActive ? 'auto' : 'none',
              }}
            >
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                {benefit.title}
              </Typography>
              <Typography variant="body2" sx={{ opacity: 0.9 }}>
                {benefit.description}
              </Typography>
            </Paper>
          );
        })}
      </Box>

      <IconButton 
        onClick={handleNext}
        sx={{ 
          position: 'absolute',
          right: 0,
          zIndex: 2,
          color: 'primary.main',
          '&:hover': { backgroundColor: 'rgba(63, 81, 181, 0.1)' }
        }}
      >
        <ArrowForwardIcon />
      </IconButton>
    </Box>
  );
};

export default BenefitsCarousel; 