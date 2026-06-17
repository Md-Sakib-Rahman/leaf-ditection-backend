import { runDetection } from "./python.service.js";
import { deleteLocalFile } from "../utils/fileCleaner.js";
import ApiError from "../utils/ApiError.js";

export const processImages = async (files) => {
  if (!files || files.length === 0) {
    throw new ApiError(400, "No images uploaded");
  }

  const results = [];

  for (const file of files) {
    try {
      // Run Python detection
      const prediction = await runDetection(file.path);

      results.push({
        filename: file.originalname,
        ...prediction,
      });
    } catch (error) {
      results.push({
        filename: file.originalname,
        success: false,
        error: error.message,
      });
    } finally {
      // Always clean up temp file
      deleteLocalFile(file.path);
    }
  }

  return {
    count: results.length,
    results,
  };
};