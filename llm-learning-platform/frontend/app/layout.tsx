import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LLM Learning Platform - Master AI and Machine Learning",
  description: "Interactive platform for learning about Language Models, Transformers, and AI. Build, train, and experiment with neural networks.",
  keywords: ["LLM", "AI", "Machine Learning", "Transformers", "Neural Networks", "Deep Learning"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
