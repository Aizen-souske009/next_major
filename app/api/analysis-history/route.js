import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';
import { getUserFromRequest } from '@/lib/auth-helper';

export async function GET(req) {
  try {
    // Get authenticated user
    const user = await getUserFromRequest(req);
    if (!user) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }

    // Get query parameters
    const { searchParams } = new URL(req.url);
    const page = parseInt(searchParams.get('page') || '1');
    const limit = parseInt(searchParams.get('limit') || '10');
    const filter = searchParams.get('filter'); // 'ai', 'real', or null for all
    const sortBy = searchParams.get('sortBy') || 'scannedAt';
    const sortOrder = searchParams.get('sortOrder') || 'desc';

    // Build where clause
    const where = {
      userId: user.id,
      ...(filter && filter !== 'all' && {
        isAI: filter === 'ai'
      })
    };

    // Get total count
    const totalCount = await prisma.imageScan.count({ where });

    // Get paginated results
    const imageScans = await prisma.imageScan.findMany({
      where,
      orderBy: {
        [sortBy]: sortOrder
      },
      skip: (page - 1) * limit,
      take: limit,
      select: {
        id: true,
        filename: true,
        originalName: true,
        sourceUrl: true,
        mimeType: true,
        sizeBytes: true,
        prediction: true,
        confidence: true,
        isAI: true,
        label: true,
        scannedAt: true,
        metadata: true
      }
    });

    // Calculate statistics
    const stats = await prisma.imageScan.groupBy({
      by: ['isAI'],
      where: { userId: user.id },
      _count: {
        id: true
      }
    });

    const aiCount = stats.find(s => s.isAI)?._count.id || 0;
    const realCount = stats.find(s => !s.isAI)?._count.id || 0;

    return NextResponse.json({
      success: true,
      data: imageScans,
      pagination: {
        page,
        limit,
        totalCount,
        totalPages: Math.ceil(totalCount / limit),
        hasNext: page < Math.ceil(totalCount / limit),
        hasPrev: page > 1
      },
      statistics: {
        total: totalCount,
        aiImages: aiCount,
        realImages: realCount,
        aiPercentage: totalCount > 0 ? ((aiCount / totalCount) * 100).toFixed(1) : 0,
        realPercentage: totalCount > 0 ? ((realCount / totalCount) * 100).toFixed(1) : 0
      }
    });
  } catch (error) {
    console.error('Error fetching analysis history:', error);
    return NextResponse.json(
      { error: 'Failed to fetch analysis history', details: error.message },
      { status: 500 }
    );
  }
}

export async function DELETE(req) {
  try {
    // Get authenticated user
    const user = await getUserFromRequest(req);
    if (!user) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }

    const { searchParams } = new URL(req.url);
    const scanId = searchParams.get('id');

    if (!scanId) {
      return NextResponse.json(
        { error: 'Scan ID is required' },
        { status: 400 }
      );
    }

    // Delete the scan (only if it belongs to the user)
    const deletedScan = await prisma.imageScan.deleteMany({
      where: {
        id: scanId,
        userId: user.id
      }
    });

    if (deletedScan.count === 0) {
      return NextResponse.json(
        { error: 'Scan not found or access denied' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      message: 'Analysis deleted successfully'
    });
  } catch (error) {
    console.error('Error deleting analysis:', error);
    return NextResponse.json(
      { error: 'Failed to delete analysis', details: error.message },
      { status: 500 }
    );
  }
}