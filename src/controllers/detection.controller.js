import asyncHandler from "../utils/asyncHandler.js";
import ApiResponse from "../utils/ApiResponse.js";
import ApiError from "../utils/ApiError.js";
import { processImages } from "../services/detection.service.js";

export const detectDisease = asyncHandler(async (req, res) => {
  if (!req.files || req.files.length === 0) {
    throw new ApiError(400, "Please upload at least one image");
  }

  const detectionResult = await processImages(req.files);

  return res.status(200).json(
    new ApiResponse(
      200,
      detectionResult,
      "Disease detection completed successfully"
    )
  );
});