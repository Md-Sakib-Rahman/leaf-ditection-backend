import { spawn } from "child_process";
import path from "path";
import ApiError from "../utils/ApiError.js";

export const runDetection = (imagePath) => {
  return new Promise((resolve, reject) => {
    const pythonScript = path.resolve("python/detect.py");

    const pythonProcess = spawn("python3", [
      pythonScript,
      imagePath,
    ]);

    let result = "";
    let errorOutput = "";

    // Capture stdout
    pythonProcess.stdout.on("data", (data) => {
      result += data.toString();
    });

    // Capture stderr
    pythonProcess.stderr.on("data", (data) => {
      errorOutput += data.toString();
    });

    // Process finished
    pythonProcess.on("close", (code) => {
      if (code !== 0) {
        return reject(
          new ApiError(
            500,
            `Python process failed: ${errorOutput}`
          )
        );
      }

      try {
        const parsedResult = JSON.parse(result);
        resolve(parsedResult);
      } catch (error) {
        reject(
          new ApiError(
            500,
            "Invalid JSON received from Python"
          )
        );
      }
    });

    pythonProcess.on("error", (error) => {
      reject(
        new ApiError(
          500,
          `Failed to start Python process: ${error.message}`
        )
      );
    });
  });
};