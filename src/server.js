import dotenv from "dotenv";
import app from "./app.js";

dotenv.config();

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log("=================================");
  console.log(" Leaf Disease Detection API");
  console.log(` Running on port: ${PORT}`);
  console.log("=================================");
});