import fs from "fs/promises";

export const deleteLocalFile = async (filePath) => {
  try {
    if (!filePath) return;

    await fs.unlink(filePath);
    console.log(`🧹 Deleted local file: ${filePath}`);
  } catch (error) {
    console.error(`❌ Failed to delete file: ${filePath}`);
    console.error(error.message);
  }
};