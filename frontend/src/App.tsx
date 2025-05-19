import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import Header from './components/layout/header';
import Home from './pages/home';
import Detect from './pages/detect';
import PaDIM from './pages/padim-information';
import STFPM from './pages/stfpm-information';
import EfficientAD from './pages/efficientad-information';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1a237e',
    },
    secondary: {
      main: '#2196F3',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
});

const App = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Header />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/detect" element={<Detect />} />
          <Route path="/padim" element={<PaDIM />} />
          <Route path="/stfpm" element={<STFPM />} />
          <Route path="/efficientad" element={<EfficientAD />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
};

export default App; 