'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { AlertCircle, CheckCircle, Eye, Calendar } from 'lucide-react';
import Image from 'next/image';

export default function RecentAnalyses() {
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchRecentAnalyses = async () => {
    try {
      const response = await fetch('/api/analysis-history?limit=6&sortBy=scannedAt&sortOrder=desc');
      const data = await response.json();

      if (data.success) {
        setAnalyses(data.data);
      } else {
        console.error('Failed to fetch recent analyses:', data.error);
      }
    } catch (error) {
      console.error('Error fetching recent analyses:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecentAnalyses();
  }, []);

  const getResultColor = (isAI) => {
    return isAI ? 'text-red-600 bg-red-50 border-red-200' : 'text-green-600 bg-green-50 border-green-200';
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return 'text-green-600';
    if (confidence >= 0.7) return 'text-yellow-600';
    return 'text-red-600';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-32">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600"></div>
      </div>
    );
  }

  if (analyses.length === 0) {
    return (
      <div className="text-center py-8">
        <div className="mx-auto h-16 w-16 text-gray-400 mb-4">
          <Calendar className="h-full w-full" />
        </div>
        <h3 className="text-lg font-medium text-gray-900 mb-2">No analyses yet</h3>
        <p className="text-gray-600 mb-4">
          Upload your first image to start detecting AI-generated content
        </p>
        <Button 
          onClick={() => window.location.href = '/dashboard/analyze'}
          className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
        >
          Start Analysis
        </Button>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {analyses.map((analysis) => (
        <div key={analysis.id} className="bg-white rounded-lg border shadow-sm hover:shadow-md transition-shadow">
          {/* Image Preview */}
          <div className="relative h-32 bg-gray-100 rounded-t-lg overflow-hidden">
            <Image
              src={analysis.sourceUrl}
              alt={analysis.originalName || 'Analysis'}
              fill
              className="object-cover"
              onError={(e) => {
                e.target.style.display = 'none';
              }}
            />
            <div className="absolute top-2 right-2">
              <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getResultColor(analysis.isAI)}`}>
                {analysis.prediction}
              </span>
            </div>
          </div>
          
          {/* Content */}
          <div className="p-3">
            <h4 className="font-medium text-gray-900 truncate text-sm" title={analysis.originalName}>
              {analysis.originalName || analysis.filename}
            </h4>
            
            <div className="mt-2 flex items-center justify-between">
              <div className="flex items-center">
                {analysis.isAI ? (
                  <AlertCircle className="h-4 w-4 text-red-600 mr-1" />
                ) : (
                  <CheckCircle className="h-4 w-4 text-green-600 mr-1" />
                )}
                <span className={`text-sm font-medium ${getConfidenceColor(analysis.confidence)}`}>
                  {(analysis.confidence * 100).toFixed(1)}%
                </span>
              </div>
              
              <span className="text-xs text-gray-500">
                {new Date(analysis.scannedAt).toLocaleDateString()}
              </span>
            </div>
            
            <div className="mt-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => window.open(analysis.sourceUrl, '_blank')}
                className="w-full text-xs"
              >
                <Eye className="h-3 w-3 mr-1" />
                View Details
              </Button>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}