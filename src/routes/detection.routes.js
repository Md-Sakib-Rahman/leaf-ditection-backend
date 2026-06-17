import { Router } from "express";
import { detectDisease } from "../controllers/detection.controller.js";
import { uploadImages } from "../middlewares/upload.middleware.js";

const router = Router();

router.post(
  "/detect",
  uploadImages,
  detectDisease
);

export default router;