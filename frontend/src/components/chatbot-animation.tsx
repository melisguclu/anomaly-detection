import { useEffect, useRef, useState } from 'react';
import { Box, Paper, Typography, IconButton, Avatar, Fade, keyframes } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import SendIcon from '@mui/icons-material/Send';
import InsertPhotoIcon from '@mui/icons-material/InsertPhoto';

// Keyframe animations
const pulseAndScale = keyframes`
  0% {
    transform: scale(1);
    box-shadow: 0 0 0 0 rgba(26, 35, 126, 0.4);
  }
  50% {
    transform: scale(0.92);
    box-shadow: 0 0 0 8px rgba(26, 35, 126, 0);
  }
  100% {
    transform: scale(1);
    box-shadow: 0 0 0 0 rgba(26, 35, 126, 0);
  }
`;

const ANIMATION_TIMINGS = {
  inputTyping: 1200,
  sendMessage: 800,
  file: 1800,
  send: 1200,
  thinking: 1500,
  response: 1800,
};

const promptText = 'Can you detect anomalies in this wood surface?';
const anomalyScore = 'Anomaly Score: 0.87';

// Height calculations for different states
const HEIGHTS = {
  initial: 80, // Just input area
  withMessage: 200, // Input + first message
  withImage: 400, // Input + message + wood image (increased for full visibility)
  withResponse: 500, // Full height with response
};

const ChatbotAnimation = () => {
  const [step, setStep] = useState(0);
  const [inputText, setInputText] = useState('');
  const [messageText, setMessageText] = useState('');
  const [isFileButtonAnimating, setIsFileButtonAnimating] = useState(false);
  const [isSendButtonAnimating, setIsSendButtonAnimating] = useState(false);
  const [showChatArea, setShowChatArea] = useState(false);
  const [fadeOut, setFadeOut] = useState(false);
  const [thinkingDots, setThinkingDots] = useState('');
  const typingInterval = useRef<number | null>(null);
  const resetTimeout = useRef<number | null>(null);

  // Calculate current height based on step
  const getCurrentHeight = () => {
    if (!showChatArea) return 0;
    if (step >= 6) return HEIGHTS.withResponse;
    if (step >= 3) return HEIGHTS.withImage;
    if (step >= 1) return HEIGHTS.withMessage;
    return HEIGHTS.initial;
  };

  // Input typing animation
  useEffect(() => {
    if (step === 0) {
      let i = 0;
      typingInterval.current = window.setInterval(() => {
        setInputText(promptText.slice(0, i + 1));
        i++;
        if (i === promptText.length) {
          clearInterval(typingInterval.current!);
          setTimeout(() => {
            setStep(1);
            setIsSendButtonAnimating(true);
            setTimeout(() => setIsSendButtonAnimating(false), 600);
          }, ANIMATION_TIMINGS.inputTyping);
        }
      }, 30);
    }
    return () => clearInterval(typingInterval.current!);
  }, [step]);

  // Message typing animation
  useEffect(() => {
    if (step === 1) {
      setShowChatArea(true);
      let i = 0;
      const messageInterval = window.setInterval(() => {
        setMessageText(promptText.slice(0, i + 1));
        i++;
        if (i === promptText.length) {
          clearInterval(messageInterval);
          setTimeout(() => {
            setStep(2);
            setIsFileButtonAnimating(true);
            setTimeout(() => setIsFileButtonAnimating(false), 600);
          }, ANIMATION_TIMINGS.sendMessage);
        }
      }, 30);
      return () => clearInterval(messageInterval);
    }
  }, [step]);

  // Step transitions
  useEffect(() => {
    if (step === 2) {
      setTimeout(() => setStep(3), ANIMATION_TIMINGS.file);
    } else if (step === 3) {
      setTimeout(() => {
        setStep(4);
        setIsSendButtonAnimating(true);
        setTimeout(() => setIsSendButtonAnimating(false), 600);
      }, ANIMATION_TIMINGS.send);
    } else if (step === 4) {
      // Start thinking animation
      setStep(5);
      let dots = '';
      const thinkingInterval = setInterval(() => {
        dots = dots.length >= 3 ? '' : dots + '.';
        setThinkingDots(dots);
      }, 400);
      
      setTimeout(() => {
        clearInterval(thinkingInterval);
        setStep(6);
      }, ANIMATION_TIMINGS.thinking);
    }
  }, [step]);

  // Repeat animation after completion
  useEffect(() => {
    if (step === 6) {
      resetTimeout.current = window.setTimeout(() => {
        // First fade out the messages
        setFadeOut(true);
        
        // After messages have faded out, collapse the container
        setTimeout(() => {
          setShowChatArea(false);
        }, 600);

        // After both fade and collapse are complete, reset the state
        setTimeout(() => {
          setInputText('');
          setMessageText('');
          setStep(0);
          setFadeOut(false);
        }, 1200); // Wait for both fade and collapse to complete
      }, 5000);
    }
    return () => {
      if (resetTimeout.current) clearTimeout(resetTimeout.current);
    };
  }, [step]);

  return (
    <Box sx={{ width: '100%', display: 'flex', justifyContent: 'center', mt: 6, mb: 4 }}>
      <Paper 
        elevation={6} 
        sx={{ 
          width: '80%', 
          maxWidth: 800, 
          borderRadius: 4, 
          overflow: 'hidden', 
          bgcolor: '#f5f7fa',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'flex-end',
        }}
      >
        {/* Chat area */}
        <Box 
          sx={{ 
            height: getCurrentHeight(),
            padding: showChatArea ? 4 : 0,
            opacity: showChatArea ? 1 : 0,
            pointerEvents: showChatArea ? 'auto' : 'none',
            transition: theme => ({
              height: theme.transitions.create('height', {
                duration: 600,
                easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
              }),
              padding: theme.transitions.create('padding', {
                duration: 600,
                easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
              }),
              opacity: theme.transitions.create('opacity', {
                duration: 600,
                easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
              }),
            }),
            display: 'flex', 
            flexDirection: 'column', 
            gap: 3,
            overflow: 'hidden',
            transformOrigin: 'top',
          }}
        >
          {/* Messages container */}
          <Box sx={{ 
            flex: 1, 
            display: 'flex', 
            flexDirection: 'column', 
            gap: 3, 
            justifyContent: 'flex-end',
            visibility: showChatArea ? 'visible' : 'hidden',
            opacity: fadeOut ? 0 : (showChatArea ? 1 : 0),
            transform: showChatArea ? 'translateY(0) scale(1)' : 'translateY(10px) scale(0.98)',
            transition: theme => ({
              opacity: theme.transitions.create('opacity', {
                duration: 600,
                easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
              }),
              transform: theme.transitions.create('transform', {
                duration: 600,
                easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
              }),
            }),
            transformOrigin: 'bottom',
          }}>
            {/* User prompt */}
            <Fade in={step >= 1 && !fadeOut} timeout={600}>
              <Box sx={{ display: 'flex', alignItems: 'flex-end', gap: 2 }}>
                <Avatar sx={{ bgcolor: '#1a237e', width: 40, height: 40, fontSize: 20 }}>U</Avatar>
                <Paper sx={{ p: 2, px: 3, bgcolor: '#e3e7f1', borderRadius: 3, maxWidth: 400 }}>
                  <Typography variant="body1" sx={{ wordBreak: 'break-word', fontSize: '1.1rem' }}>
                    {messageText}
                  </Typography>
                </Paper>
              </Box>
            </Fade>
            {/* File upload animation */}
            <Fade in={step >= 3 && !fadeOut} timeout={600}>
              <Box sx={{ display: 'flex', alignItems: 'flex-end', gap: 2 }}>
                <Avatar sx={{ bgcolor: '#1a237e', width: 40, height: 40, fontSize: 20 }}>U</Avatar>
                <Paper sx={{ p: 2, bgcolor: '#e3e7f1', borderRadius: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
                  <InsertPhotoIcon sx={{ color: '#1a237e', fontSize: 28 }} />
                  <img 
                    src="/wood-image.jpeg" 
                    alt="Wood" 
                    style={{ 
                      width: step >= 5 ? 80 : 200, 
                      borderRadius: 8,
                      transition: 'width 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
                    }} 
                  />
                </Paper>
              </Box>
            </Fade>
            {/* Thinking animation and Result container */}
            <Box sx={{ display: 'flex', alignItems: 'flex-end', gap: 2, alignSelf: 'flex-end' }}>
              <Paper 
                sx={{ 
                  p: step >= 6 ? 3 : 2,
                  bgcolor: '#fff', 
                  borderRadius: 3, 
                  border: '1px solid #e3e7f1',
                  position: 'relative',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: step >= 6 ? '100%' : '120px',
                  maxWidth: step >= 6 ? 500 : '120px',
                  height: step >= 6 ? 'auto' : 40,
                  transition: 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
                }}
              >
                {/* Thinking dots */}
                <Fade in={step === 5 && !fadeOut} timeout={600}>
                  <Box sx={{ 
                    position: 'absolute',
                    display: 'flex',
                    gap: 1,
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '100%',
                    height: '100%',
                  }}>
                    {[0, 1, 2].map((index) => (
                      <Box
                        key={index}
                        sx={{
                          width: 8,
                          height: 8,
                          borderRadius: '50%',
                          bgcolor: '#1a237e',
                          opacity: thinkingDots.length > index ? 1 : 0.3,
                          transition: 'opacity 0.2s ease-in-out',
                        }}
                      />
                    ))}
                  </Box>
                </Fade>

                {/* Result content */}
                <Fade in={step >= 6 && !fadeOut} timeout={800}>
                  <Box sx={{ 
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: 'center', 
                    gap: 2,
                    width: '100%',
                  }}>
                    <img src="/result.jpeg" alt="Result" style={{ width: '100%', borderRadius: 8 }} />
                    <Typography variant="h6" sx={{ color: '#1a237e', fontWeight: 600 }}>{anomalyScore}</Typography>
                  </Box>
                </Fade>
              </Paper>
              <Avatar sx={{ bgcolor: '#e3e7f1', color: '#1a237e', width: 40, height: 40, fontSize: 20 }}>AI</Avatar>
            </Box>
          </Box>
        </Box>
        {/* Input area (fake, for animation only) */}
        <Box sx={{ display: 'flex', alignItems: 'center', px: 3, py: 2, borderTop: '1px solid #e3e7f1', bgcolor: '#f8fafc' }}>
          <Box sx={{ flex: 1, mr: 2 }}>
            <Paper sx={{ p: 2, px: 3, borderRadius: 2, bgcolor: '#fff', border: '1px solid #e3e7f1', color: '#888', fontSize: '1.1rem' }}>
              {inputText}
            </Paper>
          </Box>
          <IconButton
            sx={{
              width: 48,
              height: 48,
              bgcolor: step === 2 ? '#e3e7f1' : 'transparent',
              border: step === 2 ? '2px solid #1a237e' : 'none',
              transition: 'all 0.2s',
              mr: 2,
              animation: isFileButtonAnimating ? `${pulseAndScale} 0.6s ease-in-out` : 'none',
              '&:hover': {
                bgcolor: step === 2 ? '#e3e7f1' : 'transparent',
              },
            }}
            disabled
          >
            <AddIcon sx={{ 
              color: '#1a237e', 
              fontSize: 28,
              transform: isFileButtonAnimating ? 'scale(0.85)' : 'scale(1)',
              transition: 'transform 0.2s ease-in-out',
            }} />
          </IconButton>
          <IconButton
            sx={{
              width: 48,
              height: 48,
              bgcolor: step === 4 ? '#1a237e' : '#e3e7f1',
              color: step === 4 ? 'white' : '#1a237e',
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
              animation: isSendButtonAnimating ? `${pulseAndScale} 0.6s ease-in-out` : 'none',
              '&:hover': {
                bgcolor: step === 4 ? '#1a237e' : '#e3e7f1',
              },
            }}
            disabled
          >
            <SendIcon sx={{ 
              fontSize: 24,
              transform: isSendButtonAnimating ? 'scale(0.85) translateX(2px)' : 'scale(1) translateX(0)',
              transition: 'transform 0.2s ease-in-out',
            }} />
          </IconButton>
        </Box>
      </Paper>
    </Box>
  );
};

export default ChatbotAnimation; 