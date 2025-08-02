'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Button } from '@/components/ui/button';
import { Upload, AlertCircle, CheckCircle, Loader2 } from 'lucide-react';
import Image from 'next/image';

export default function DeepfakeDetector() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.webp']
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024 // 10MB
  });

  const analyzeImage = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);

      const response = await fetch('/api/deepfake', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to analyze image');
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const resetAnalysis = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const getResultColor = (isAI) => {
    return isAI ? 'text-red-600 bg-red-50 border-red-200' : 'text-green-600 bg-green-50 border-green-200';
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return 'text-green-600';
    if (confidence >= 0.7) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
          DeepTrace AI Detector
        </h1>
        <p className="text-gray-600 mt-2">
          Upload an image to detect if it's AI-generated or real
        </p>
      </div>

      {/* Upload Area */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
          isDragActive
            ? 'border-blue-400 bg-blue-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
      >
        <input {...getInputProps()} />
        <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        {isDragActive ? (
          <p className="text-blue-600">Drop the image here...</p>
        ) : (
          <div>
            <p className="text-gray-600 mb-2">
              Drag & drop an image here, or click to select
            </p>
            <p className="text-sm text-gray-400">
              Supports JPEG, PNG, WebP (max 10MB)
            </p>
          </div>
        )}
      </div>

      {/* Preview */}
      {preview && (
        <div className="bg-white rounded-lg border p-6">
          <h3 className="text-lg font-semibold mb-4">Image Preview</h3>
          <div className="flex flex-col md:flex-row gap-6">
            <div className="flex-1">
              <div className="relative w-full h-64 bg-gray-100 rounded-lg overflow-hidden">
                <Image
                  src={preview}
                  alt="Preview"
                  fill
                  className="object-contain"
                />
              </div>
            </div>
            <div className="flex-1 space-y-4">
              <div>
                <p className="text-sm text-gray-600">File: {selectedFile?.name}</p>
                <p className="text-sm text-gray-600">
                  Size: {(selectedFile?.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              
              <Button
                onClick={analyzeImage}
                disabled={loading}
                className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  'Analyze Image'
                )}
              </Button>

              <Button
                onClick={resetAnalysis}
                variant="outline"
                className="w-full"
              >
                Upload New Image
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertCircle className="h-5 w-5 text-red-600 mr-2" />
            <p className="text-red-800">{error}</p>
          </div>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="bg-white rounded-lg border p-6">
          <h3 className="text-lg font-semibold mb-4">Analysis Results</h3>
          
          <div className="space-y-4">
            {/* Main Result */}
            <div className={`p-4 rounded-lg border ${getResultColor(result.isAI)}`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  {result.isAI ? (
                    <AlertCircle className="h-6 w-6 mr-2" />
                  ) : (
                    <CheckCircle className="h-6 w-6 mr-2" />
                  )}
                  <div>
                    <p className="font-semibold text-lg">
                      {result.isAI ? 'AI-Generated' : 'Real Image'}
                    </p>
                    <p className="text-sm opacity-80">
                      {result.isAI 
                        ? 'This image appears to be artificially generated'
                        : 'This image appears to be authentic'
                      }
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className={`text-2xl font-bold ${getConfidenceColor(result.confidence)}`}>
                    {(result.confidence * 100).toFixed(1)}%
                  </p>
                  <p className="text-sm text-gray-600">Confidence</p>
                </div>
              </div>
            </div>

            {/* Confidence Meter */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Confidence Level</span>
                <span className={getConfidenceColor(result.confidence)}>
                  {result.confidence >= 0.9 ? 'Very High' : 
                   result.confidence >= 0.7 ? 'High' : 'Moderate'}
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-500 ${
                    result.confidence >= 0.9 ? 'bg-green-500' :
                    result.confidence >= 0.7 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${result.confidence * 100}%` }}
                />
              </div>
            </div>

            {/* Additional Info */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-medium mb-2">Detection Details</h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">Prediction:</span>
                  <span className="ml-2 font-medium">{result.prediction}</span>
                </div>
                <div>
                  <span className="text-gray-600">Model:</span>
                  <span className="ml-2 font-medium">ResNet50</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}