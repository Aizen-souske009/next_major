import NextAuth from "next-auth";
import GoogleProvider from "next-auth/providers/google";
import GitHubProvider from "next-auth/providers/github";
import { PrismaAdapter } from "@next-auth/prisma-adapter";
import { prisma } from "@/lib/prisma";

const handler = NextAuth({
  adapter: PrismaAdapter(prisma),
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    }),
    GitHubProvider({
      clientId: process.env.GITHUB_CLIENT_ID,
      clientSecret: process.env.GITHUB_CLIENT_SECRET,
    }),
  ],
  secret: process.env.NEXTAUTH_SECRET,
  session: { strategy: "database" },
  callbacks: {
    async signIn({ user, account, profile }) {
      // Let the adapter handle user creation, just return true
      return true;
    },
    async session({ session, user }) {
      // Add user info to session
      if (session?.user && user) {
        session.user.id = user.id;
        session.user.role = user.role;
        session.user.lastLogin = user.lastLogin;
        
        // Update lastLogin timestamp
        try {
          await prisma.user.update({
            where: { id: user.id },
            data: { lastLogin: new Date() }
          });
        } catch (error) {
          console.error("Error updating lastLogin:", error);
        }
      }
      return session;
    },
  },
});

export { handler as GET, handler as POST };
