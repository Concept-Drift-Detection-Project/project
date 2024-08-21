import React, { useState } from 'react';
import { Container, Typography, Button, Checkbox, FormControlLabel, FormGroup, Box } from '@mui/material';
import { UploadFile } from '@mui/icons-material';


function Home() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedModels, setSelectedModels] = useState({
    linearRegression: false,
    decisionTree: false,
    randomForest: false,
    supportVector: false,
  });

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleModelChange = (event) => {
    setSelectedModels({
      ...selectedModels,
      [event.target.name]: event.target.checked,
    });
  };

  const handleSubmit = () => {
    // Implement form submission logic, possibly including file upload and selected models
    console.log("Selected File:", selectedFile);
    console.log("Selected Models:", selectedModels);
  };

  return (
    <Container maxWidth="sm">
      <Typography variant="h4" component="h1" gutterBottom>
        Regression Model Selector
      </Typography>

      <Box my={4}>
        <Button
          variant="contained"
          component="label"
          startIcon={<UploadFile />}
        >
          Upload CSV
          <input
            type="file"
            accept=".csv"
            hidden
            onChange={handleFileChange}
          />
        </Button>
        {selectedFile && <Typography>{selectedFile.name}</Typography>}
      </Box>

      <FormGroup>
        <Typography variant="h6" gutterBottom>
          Select Regression Models:
        </Typography>
        <FormControlLabel
          control={<Checkbox checked={selectedModels.linearRegression} onChange={handleModelChange} name="linearRegression" />}
          label="Linear Regression"
        />
        <FormControlLabel
          control={<Checkbox checked={selectedModels.decisionTree} onChange={handleModelChange} name="decisionTree" />}
          label="Decision Tree Regression"
        />
        <FormControlLabel
          control={<Checkbox checked={selectedModels.randomForest} onChange={handleModelChange} name="randomForest" />}
          label="RandomForest Regression"
        />
        <FormControlLabel
          control={<Checkbox checked={selectedModels.supportVector} onChange={handleModelChange} name="supportVector" />}
          label="Support Vector Regression"
        />
      </FormGroup>

      <Box my={4}>
        <Button variant="contained" color="primary" onClick={handleSubmit}>
          Submit
        </Button>
      </Box>
    </Container>
  );
}

export default Home;
