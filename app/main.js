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

// Function to get all files recursively
const getAllFiles = (dirPath, arrayOfFiles = []) => {
  try {
    const files = fs.readdirSync(dirPath);

    files.forEach(file => {
      const fullPath = path.join(dirPath, file);
      try {
        if (fs.statSync(fullPath).isDirectory()) {
          arrayOfFiles = getAllFiles(fullPath, arrayOfFiles); // Recurse into subdirectory
        } else {
          arrayOfFiles.push(fullPath); // Add file to array
        }
      } catch (error) {
        console.warn(`Skipping file/directory: ${fullPath}`); // Handle permission or access errors gracefully
      }
    });
  } catch (error) {
    console.warn(`Skipping directory: ${dirPath}`); // Handle top-level directory errors gracefully
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
