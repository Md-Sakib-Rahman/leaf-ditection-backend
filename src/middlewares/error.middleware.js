import ApiError from "../utils/ApiError.js";

const errorMiddleware = (err, req, res, next) => {
  // Default values
  let statusCode = err.statusCode || 500;
  let message = err.message || "Internal Server Error";
  let errors = err.errors || [];

  // Handle non-ApiError exceptions
  if (!(err instanceof ApiError)) {
    console.error("❌ Unexpected Error:", err);

    statusCode = 500;
    message = "Internal Server Error";
    errors = [];
  }

  return res.status(statusCode).json({
    success: false,
    statusCode,
    message,
    errors,
    data: null,
  });
};

export default errorMiddleware;