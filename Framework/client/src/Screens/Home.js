import React, { useState } from 'react';
import { Container, Typography, Button, Checkbox, FormControlLabel, FormGroup, Box, Paper } from '@mui/material';
import { UploadFile } from '@mui/icons-material';
import { blue, green, purple } from '@mui/material/colors';

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
    console.log("Selected File:", selectedFile);
    console.log("Selected Models:", selectedModels);
  };

  return (
    <Container maxWidth="sm">
      <Paper elevation={3} style={{ padding: '20px', backgroundColor: '#f3f4f6' }}>
        <Typography 
          variant="h3" 
          component="h1" 
          gutterBottom 
          align="center" 
          color={blue[700]}
          style={{ fontWeight: 'bold' }}
        >
          Concept Drift Detector
        </Typography>

        <Box my={4} textAlign="center">
          <Button
            variant="contained"
            component="label"
            startIcon={<UploadFile />}
            sx={{
              backgroundColor: green[500],
              color: 'white',
              '&:hover': {
                backgroundColor: green[700],
              },
              padding: '10px 20px',
            }}
          >
            Upload CSV
            <input
              type="file"
              accept=".csv"
              hidden
              onChange={handleFileChange}
            />
          </Button>
          {selectedFile && (
            <Typography 
              mt={2} 
              color="textSecondary"
              style={{ fontStyle: 'italic' }}
            >
              {selectedFile.name}
            </Typography>
          )}
        </Box>

        <FormGroup>
          <Typography 
            variant="h5" 
            gutterBottom 
            align="center" 
            color={purple[500]}
          >
            Select Regression Models:
          </Typography>
          <FormControlLabel
            control={<Checkbox checked={selectedModels.linearRegression} onChange={handleModelChange} name="linearRegression" sx={{ color: blue[600] }} />}
            label="Linear Regression"
          />
          <FormControlLabel
            control={<Checkbox checked={selectedModels.decisionTree} onChange={handleModelChange} name="decisionTree" sx={{ color: blue[600] }} />}
            label="Decision Tree Regression"
          />
          <FormControlLabel
            control={<Checkbox checked={selectedModels.randomForest} onChange={handleModelChange} name="randomForest" sx={{ color: blue[600] }} />}
            label="RandomForest Regression"
          />
          <FormControlLabel
            control={<Checkbox checked={selectedModels.supportVector} onChange={handleModelChange} name="supportVector" sx={{ color: blue[600] }} />}
            label="Support Vector Regression"
          />
        </FormGroup>

        <Box my={4} textAlign="center">
          <Button 
            variant="contained" 
            color="primary" 
            onClick={handleSubmit}
            sx={{
              backgroundColor: purple[600],
              color: 'white',
              '&:hover': {
                backgroundColor: purple[800],
              },
              padding: '10px 20px',
            }}
          >
            Submit
          </Button>
        </Box>
      </Paper>
    </Container>
  );
}

export default Home;
