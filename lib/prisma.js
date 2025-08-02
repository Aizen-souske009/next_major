import { PrismaClient } from "@prisma/client";

// Use a global variable to avoid multiple Prisma Client instances in development
const globalForPrisma = globalThis;

export const prisma =
  globalForPrisma.prisma ||
  new PrismaClient({
    log: ["query"], // Optional: shows all queries in the console (good for debugging)
  });

if (process.env.NODE_ENV !== "production") globalForPrisma.prisma = prisma;

export default prisma;
