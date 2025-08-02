# next_major

> A full-stack Next.js 15 application with role-based authentication (NextAuth.js + Prisma + PostgreSQL) and a companion deepfake detection training script.

## 🚀 Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Tech Stack](#tech-stack)  
4. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Installation](#installation)  
   - [Environment Variables](#environment-variables)  
   - [Database Setup](#database-setup)  
   - [Run the App](#run-the-app)  
5. [Deepfake Model Training](#deepfake-model-training)  
6. [Project Structure](#project-structure)  
7. [Contributing](#contributing)  
8. [License](#license)  

---

## 📋 Project Overview

**next_major** is designed as a modern Next.js application that demonstrates:

- Secure, credentials-and-OAuth authentication  
- Role-based routing (e.g. `ADMIN` vs `USER`)  
- A polished UI with Tailwind CSS & lucide-react icons  
- Seamless Prisma integration with PostgreSQL  
- A standalone Python script (`train_deepfake_model.py`) to train a ResNet50-based deepfake detector  

This boilerplate can be your “launchpad” for production apps that need both robust auth and custom ML workflows.

---

## ✨ Features

- **NextAuth.js** for email/password and social logins (GitHub, Google)  
- **JWT** sessions enriched with user roles  
- **Prisma** ORM + PostgreSQL migrations  
- **Tailwind CSS** & **Next.js 15 (app directory)**  
- **Python** script for training a deepfake detection model  
- Environment-driven config for safe secrets management  

---

## 🛠 Tech Stack

- **Frontend / Backend**  
  - Next.js 15 (App Router)  
  - NextAuth.js  
  - React + lucide-react icons  
  - Tailwind CSS  

- **Database**  
  - PostgreSQL  
  - Prisma ORM  

- **ML Training**  
  - Python 3.x  
  - TensorFlow / PyTorch (depending on your `train_deepfake_model.py` setup)  
  - `train_deepfake_model.py` with ResNet50 + custom CNN head  

---

## 🔧 Getting Started

### Prerequisites

- Node.js ≥ 18  
- pnpm / npm / Yarn  
- Python 3.8+  
- PostgreSQL  

### Installation

1. Clone the repo  
   ```bash
   git clone https://github.com/Aizen-souske009/next_major.git
   cd next_major
   ```

2. Install JS dependencies
   ```bash
   npm i
   ```
# Database
DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/DBNAME

# NextAuth.js
NEXTAUTH_SECRET=some-very-strong-secret
NEXTAUTH_URL=http://localhost:3000

# OAuth Providers
GITHUB_ID=your_github_client_id
GITHUB_SECRET=your_github_client_secret
GOOGLE_ID=your_google_client_id
GOOGLE_SECRET=your_google_client_secret

Database Setup
1. Generate Prisma client & run migrations:
   ```bash
   npx prisma migrate dev --name init
   npx prisma generate
   ```
2. (Optional) Open Prisma Studio:
   ```bash
   npx prisma studio
   ```
Run the App
   ```bash
   npm run dev
   ```
