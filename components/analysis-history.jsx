'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { 
  Search, 
  Filter, 
  Trash2, 
  Download, 
  Eye, 
  Calendar,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import Image from 'next/image';

export default function AnalysisHistory() {
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [statistics, setStatistics] = useState(null);
  const [pagination, setPagination] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [filter, setFilter] = useState('all');
  const [sortBy, setSortBy] = useState('scannedAt');
  const [sortOrder, setSortOrder] = useState('desc');
  const [searchTerm, setSearchTerm] = useState('');

  const fetchAnalyses = async (page = 1) => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        page: page.toString(),
        limit: '12',
        filter,
        sortBy,
        sortOrder
      });

      const response = await fetch(`/api/analysis-history?${params}`);
      const data = await response.json();

      if (data.success) {
        setAnalyses(data.data);
        setStatistics(data.statistics);
        setPagination(data.pagination);
        setCurrentPage(page);
      } else {
        console.error('Failed to fetch analyses:', data.error);
      }
    } catch (error) {
      console.error('Error fetching analyses:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalyses(1);
  }, [filter, sortBy, sortOrder]);

  const handleDelete = async (scanId) => {
    if (!confirm('Are you sure you want to delete this analysis?')) return;

    try {
      const response = await fetch(`/api/analysis-history?id=${scanId}`, {
        method: 'DELETE'
      });

      const data = await response.json();
      if (data.success) {
        fetchAnalyses(currentPage);
      } else {
        alert('Failed to delete analysis: ' + data.error);
      }
    } catch (error) {
      console.error('Error deleting analysis:', error);
      alert('Failed to delete analysis');
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getResultColor = (isAI) => {
    return isAI ? 'text-red-600 bg-red-50 border-red-200' : 'text-green-600 bg-green-50 border-green-200';
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return 'text-green-600';
    if (confidence >= 0.7) return 'text-yellow-600';
    return 'text-red-600';
  };

  const filteredAnalyses = analyses.filter(analysis =>
    analysis.originalName?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    analysis.prediction?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  if (loading && analyses.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading analysis history...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Statistics Cards */}
      {statistics && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white rounded-lg border p-4">
            <div className="flex items-center">
              <TrendingUp className="h-8 w-8 text-blue-600" />
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-600">Total Scans</p>
                <p className="text-2xl font-bold text-gray-900">{statistics.total}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg border p-4">
            <div className="flex items-center">
              <AlertCircle className="h-8 w-8 text-red-600" />
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-600">AI Images</p>
                <p className="text-2xl font-bold text-gray-900">{statistics.aiImages}</p>
                <p className="text-xs text-gray-500">{statistics.aiPercentage}%</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg border p-4">
            <div className="flex items-center">
              <CheckCircle className="h-8 w-8 text-green-600" />
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-600">Real Images</p>
                <p className="text-2xl font-bold text-gray-900">{statistics.realImages}</p>
                <p className="text-xs text-gray-500">{statistics.realPercentage}%</p>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg border p-4">
            <div className="flex items-center">
              <Calendar className="h-8 w-8 text-purple-600" />
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-600">This Month</p>
                <p className="text-2xl font-bold text-gray-900">{statistics.total}</p>
                <p className="text-xs text-gray-500">All time</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="bg-white rounded-lg border p-4">
        <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
          <div className="flex items-center gap-4 flex-1">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
              <input
                type="text"
                placeholder="Search by filename or result..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 pr-4 py-2 border rounded-lg w-full focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              />
            </div>
            
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500"
            >
              <option value="all">All Results</option>
              <option value="ai">AI Images</option>
              <option value="real">Real Images</option>
            </select>
          </div>
          
          <div className="flex items-center gap-2">
            <select
              value={`${sortBy}-${sortOrder}`}
              onChange={(e) => {
                const [field, order] = e.target.value.split('-');
                setSortBy(field);
                setSortOrder(order);
              }}
              className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500"
            >
              <option value="scannedAt-desc">Newest First</option>
              <option value="scannedAt-asc">Oldest First</option>
              <option value="confidence-desc">Highest Confidence</option>
              <option value="confidence-asc">Lowest Confidence</option>
              <option value="originalName-asc">Name A-Z</option>
              <option value="originalName-desc">Name Z-A</option>
            </select>
          </div>
        </div>
      </div>

      {/* Results Grid */}
      {filteredAnalyses.length === 0 ? (
        <div className="text-center py-12">
          <div className="mx-auto h-24 w-24 text-gray-400">
            <Search className="h-full w-full" />
          </div>
          <h3 className="mt-4 text-lg font-medium text-gray-900">No analyses found</h3>
          <p className="mt-2 text-gray-600">
            {searchTerm ? 'Try adjusting your search terms' : 'Upload some images to see your analysis history'}
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {filteredAnalyses.map((analysis) => (
            <div key={analysis.id} className="bg-white rounded-lg border shadow-sm hover:shadow-md transition-shadow">
              {/* Image Preview */}
              <div className="relative h-48 bg-gray-100 rounded-t-lg overflow-hidden">
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
              <div className="p-4">
                <h3 className="font-medium text-gray-900 truncate" title={analysis.originalName}>
                  {analysis.originalName || analysis.filename}
                </h3>
                
                <div className="mt-2 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Confidence</span>
                    <span className={`text-sm font-medium ${getConfidenceColor(analysis.confidence)}`}>
                      {(analysis.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Size</span>
                    <span className="text-sm text-gray-900">
                      {formatFileSize(analysis.sizeBytes)}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Scanned</span>
                    <span className="text-sm text-gray-900">
                      {new Date(analysis.scannedAt).toLocaleDateString()}
                    </span>
                  </div>
                </div>
                
                {/* Actions */}
                <div className="mt-4 flex items-center justify-between">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => window.open(analysis.sourceUrl, '_blank')}
                  >
                    <Eye className="h-4 w-4 mr-1" />
                    View
                  </Button>
                  
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleDelete(analysis.id)}
                    className="text-red-600 hover:text-red-700 hover:bg-red-50"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {pagination && pagination.totalPages > 1 && (
        <div className="flex items-center justify-between bg-white rounded-lg border p-4">
          <div className="text-sm text-gray-600">
            Showing {((currentPage - 1) * pagination.limit) + 1} to {Math.min(currentPage * pagination.limit, pagination.totalCount)} of {pagination.totalCount} results
          </div>
          
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => fetchAnalyses(currentPage - 1)}
              disabled={!pagination.hasPrev}
            >
              <ChevronLeft className="h-4 w-4" />
              Previous
            </Button>
            
            <span className="px-3 py-1 text-sm">
              Page {currentPage} of {pagination.totalPages}
            </span>
            
            <Button
              variant="outline"
              size="sm"
              onClick={() => fetchAnalyses(currentPage + 1)}
              disabled={!pagination.hasNext}
            >
              Next
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}