import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import { writeFile } from 'fs/promises';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { prisma } from '@/lib/prisma';
import { getUserFromRequest } from '@/lib/auth-helper';

// Helper function to run Python script
async function runPythonScript(imagePath) {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [
      'validate_deepfake_model.py',
      '--model_path', 'training_output/models/best_model.pth',
      '--model_type', 'resnet',
      '--image', imagePath
    ]);

    let result = '';
    let error = '';

    pythonProcess.stdout.on('data', (data) => {
      result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python process exited with code ${code}: ${error}`));
      } else {
        // Parse the output to extract prediction and confidence
        const predictionMatch = result.match(/Prediction: (AI|Real)/);
        const confidenceMatch = result.match(/Confidence: ([0-9.]+)/);
        
        if (predictionMatch && confidenceMatch) {
          resolve({
            prediction: predictionMatch[1],
            confidence: parseFloat(confidenceMatch[1]),
            rawOutput: result
          });
        } else {
          reject(new Error('Failed to parse prediction results'));
        }
      }
    });
  });
}

export async function POST(req) {
  try {
    // Get authenticated user
    const user = await getUserFromRequest(req);
    if (!user) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }

    // Check if request is multipart form data
    if (!req.headers.get('content-type')?.includes('multipart/form-data')) {
      return NextResponse.json(
        { error: 'Request must be multipart/form-data' },
        { status: 400 }
      );
    }

    // Parse form data
    const formData = await req.formData();
    const file = formData.get('image');

    if (!file || !(file instanceof Blob)) {
      return NextResponse.json(
        { error: 'No image file provided' },
        { status: 400 }
      );
    }

    // Check file type
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      return NextResponse.json(
        { error: 'Invalid file type. Only JPEG, PNG, and WebP are supported' },
        { status: 400 }
      );
    }

    // Create a unique filename
    const buffer = Buffer.from(await file.arrayBuffer());
    const filename = `${uuidv4()}.${file.type.split('/')[1]}`;
    const uploadDir = path.join(process.cwd(), 'uploads');
    const filePath = path.join(uploadDir, filename);

    // Ensure upload directory exists
    await writeFile(filePath, buffer);

    // Process the image with the model
    const result = await runPythonScript(filePath);

    // Save analysis result to database
    const imageScan = await prisma.imageScan.create({
      data: {
        userId: user.id,
        filename: filename,
        originalName: file.name,
        sourceUrl: `/uploads/${filename}`,
        mimeType: file.type,
        sizeBytes: file.size,
        prediction: result.prediction,
        confidence: result.confidence,
        isAI: result.prediction === 'AI',
        label: result.prediction === 'AI' ? 'AI' : 'REAL',
        metadata: {
          modelType: 'resnet',
          modelPath: 'training_output/models/best_model.pth',
          analysisTimestamp: new Date().toISOString()
        }
      }
    });

    // Return the prediction with database ID
    return NextResponse.json({
      success: true,
      id: imageScan.id,
      prediction: result.prediction,
      confidence: result.confidence,
      isAI: result.prediction === 'AI',
      filename: filename,
      originalName: file.name,
      scannedAt: imageScan.scannedAt
    });
  } catch (error) {
    console.error('Error processing image:', error);
    return NextResponse.json(
      { error: 'Failed to process image', details: error.message },
      { status: 500 }
    );
  }
}