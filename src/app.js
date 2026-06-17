import express from "express";
import cors from "cors";

import detectionRoutes from "./routes/detection.routes.js";
import errorMiddleware from "./middlewares/error.middleware.js";

const app = express();

/*
|--------------------------------------------------------------------------
| Global Middlewares
|--------------------------------------------------------------------------
*/

// Enable CORS
app.use(cors());

// Parse JSON requests
app.use(express.json());

// Parse URL encoded data
app.use(express.urlencoded({ extended: true }));

/*
|--------------------------------------------------------------------------
| Health Check Route
|--------------------------------------------------------------------------
*/

app.get("/", (req, res) => {
  res.status(200).json({
    success: true,
    message: "🌿 Plant Disease Detection API is running",
  });
});

/*
|--------------------------------------------------------------------------
| API Routes
|--------------------------------------------------------------------------
*/

app.use("/api", detectionRoutes);

/*
|--------------------------------------------------------------------------
| Error Handler (must be last)
|--------------------------------------------------------------------------
*/

app.use(errorMiddleware);

export default app;