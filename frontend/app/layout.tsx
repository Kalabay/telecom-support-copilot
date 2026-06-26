import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin", "cyrillic"], display: "swap" });

export const metadata: Metadata = {
  title: "Копилот оператора — телеком-поддержка",
  description:
    "LLM-ассистент для операторов колл-центров телеком-поддержки с распознаванием эмоций клиента",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ru">
      <body className={`${inter.className} min-h-screen`}>{children}</body>
    </html>
  );
}
