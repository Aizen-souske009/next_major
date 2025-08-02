-- AlterTable
ALTER TABLE "ImageScan" ADD COLUMN     "isAI" BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN     "originalName" TEXT,
ADD COLUMN     "prediction" TEXT;

-- CreateIndex
CREATE INDEX "ImageScan_isAI_idx" ON "ImageScan"("isAI");
