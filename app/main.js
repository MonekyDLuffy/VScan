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
    },
  });

  mainWindow.loadFile('index.html');
});

// Browse for directory
ipcMain.handle('browse-directory', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
  });

  return result.filePaths[0]; // Return the selected directory path
});

// Recursive function to get all files
const getAllFiles = (dirPath, arrayOfFiles = []) => {
  const files = fs.readdirSync(dirPath);

  files.forEach(file => {
    const fullPath = path.join(dirPath, file);
    if (fs.statSync(fullPath).isDirectory()) {
      arrayOfFiles = getAllFiles(fullPath, arrayOfFiles); // Recurse into subdirectory
    } else {
      arrayOfFiles.push(fullPath); // Add file to array
    }
  });

  return arrayOfFiles;
};

// Start scanning
ipcMain.handle('start-scanning', async (event, directoryPath) => {
  if (!fs.existsSync(directoryPath)) {
    return { error: 'Directory not found!' };
  }

  const allFiles = getAllFiles(directoryPath); // Get all files recursively
  return { files: allFiles };
});
