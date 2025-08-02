import { prisma } from '@/lib/prisma';

export async function getUserFromRequest(req) {
  try {
    const cookies = req.headers.get('cookie');
    if (!cookies) {
      return null;
    }

    // Try different cookie name formats
    let sessionToken = null;
    
    // Development format
    const devMatch = cookies.match(/next-auth\.session-token=([^;]+)/);
    if (devMatch) {
      sessionToken = devMatch[1];
    }
    
    // Production format
    if (!sessionToken) {
      const prodMatch = cookies.match(/__Secure-next-auth\.session-token=([^;]+)/);
      if (prodMatch) {
        sessionToken = prodMatch[1];
      }
    }
    
    // Alternative format
    if (!sessionToken) {
      const altMatch = cookies.match(/next-auth\.session-token\.0=([^;]+)/);
      if (altMatch) {
        sessionToken = altMatch[1];
      }
    }

    if (!sessionToken) {
      console.log('No session token found in cookies:', cookies);
      return null;
    }

    // Get session from database
    const session = await prisma.session.findUnique({
      where: { sessionToken },
      include: { user: true }
    });

    if (!session || session.expires < new Date()) {
      console.log('Session not found or expired');
      return null;
    }

    return session.user;
  } catch (error) {
    console.error('Error getting user from request:', error);
    return null;
  }
}