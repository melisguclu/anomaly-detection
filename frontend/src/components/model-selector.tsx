import { FormControl, InputLabel, Select, MenuItem, SelectChangeEvent } from '@mui/material';

export type ModelType = 'padim' | 'stfpm';

interface ModelSelectorProps {
  selectedModel: ModelType;
  onModelChange: (model: ModelType) => void;
  disabled?: boolean;
}

const ModelSelector = ({ selectedModel, onModelChange, disabled = false }: ModelSelectorProps) => {
  const handleChange = (event: SelectChangeEvent<ModelType>) => {
    onModelChange(event.target.value as ModelType);
  };

  return (
    <FormControl fullWidth sx={{ mb: 2 }}>
      <InputLabel id="model-selector-label">Model</InputLabel>
      <Select
        labelId="model-selector-label"
        id="model-selector"
        value={selectedModel}
        label="Model"
        onChange={handleChange}
        disabled={disabled}
      >
        <MenuItem value="padim">PaDIM</MenuItem>
        <MenuItem value="stfpm">STFPM</MenuItem>
      </Select>
    </FormControl>
  );
};

export default ModelSelector; 