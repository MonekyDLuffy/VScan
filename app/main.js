const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const fs = require('fs');
const path = require('path');

let mainWindow;

app.on('ready', () => {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    }
  });
  mainWindow.setMenuBarVisibility(false);
  mainWindow.loadFile('index.html');
});

// Function to get all files 
const getAllFiles = (dirPath, arrayOfFiles = []) => {
  try {
    const files = fs.readdirSync(dirPath);

    files.forEach(file => {
      const fullPath = path.join(dirPath, file);
      try {
        if (fs.statSync(fullPath).isDirectory()) {
          arrayOfFiles = getAllFiles(fullPath, arrayOfFiles); //  subdirectory
        } else {
          arrayOfFiles.push(fullPath); // Add file to array
        }
      } catch (error) {
        console.warn(`Skipping file/directory: ${fullPath}`); // Handle permission or access errors 
      }
    });
  } catch (error) {
    console.warn(`Skipping directory: ${dirPath}`); // Handle top-level directory errors 
  }

  return arrayOfFiles;
};

// IPC handler for browsing directory
ipcMain.handle('browse-directory', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
  });

  return result.filePaths[0]; // Return the selected directory path
});

// IPC handler for starting the scan
ipcMain.handle('start-scanning', async (event, directoryPath) => {
  if (!fs.existsSync(directoryPath)) {
    return { error: `Directory not found: ${directoryPath}` };
  }

  const allFiles = getAllFiles(directoryPath); // Get all files recursively
  return { files: allFiles };
});

function loadMalwareDetectionModel(modelPath) {
  console.log("Initializing malware detection pipeline...");
  // loading a model
  const startTime = Date.now();
  while (Date.now() - startTime < 500);
  console.log(`Model loaded successfully from ${modelPath}`);
}

// Fake model inference
function analyzeFileWithModel(filePath) {
  console.log(`Analyzing file: ${filePath}`);
  
  // "model" processing logic
  const fileContent = readFileContent(filePath); // Read file content
  // detection result
  const isMalware = fileContent.endsWith("A"); 
  return {
    filePath,
    result: isMalware ? "malware" : "clean",
  };
}

// Simulate reading file content
function readFileContent(filePath) {
  try {
    const fileExtension = path.extname(filePath);
    console.log(`Reading file with extension: ${fileExtension}`);
    return `Fake content of ${filePath}`; // Dummy content
  } catch (error) {
    console.error(`Failed to read file: ${error.message}`);
    return "";
  }
}

function generateFakeHash(content) {
  const hashValue = Math.random().toString(36).substring(2, 8).toUpperCase();
  console.log(`Generated hash: ${hashValue}`);
  return hashValue;
}

function performScan(files) {
  console.log("Starting scan process...");
  
  loadMalwareDetectionModel("malware_pipeline.pkl"); //model loading

  const results = [];
  for (const file of files) {
    const result = analyzeFileWithModel(file);
    console.log(`Scan result for ${file}: ${result.result}`);
    results.push(result);
  }

  console.log("Scan process completed.");
  return results;
}
