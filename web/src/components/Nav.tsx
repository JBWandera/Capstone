"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links: { href: string; label: string; short?: string }[] = [
  { href: "/", label: "Detect" },
  { href: "/map", label: "Map" },
  { href: "/about", label: "How it works", short: "About" },
];

export function Logo({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="8" fill="#f5a524" />
      <path d="M9 27 L14 5 M23 27 L18 5" stroke="#1a1204" strokeWidth="2.2" strokeLinecap="round" />
      <path d="M16 8v3M16 15v3" stroke="#1a1204" strokeWidth="2" strokeLinecap="round" />
      <ellipse cx="16.5" cy="23" rx="4.2" ry="1.9" fill="#1a1204" />
    </svg>
  );
}

export function Nav() {
  const path = usePathname();
  return (
    <header className="sticky top-0 z-20 bg-bg/90 backdrop-blur border-b border-line">
      <nav className="max-w-5xl mx-auto px-4 h-14 flex items-center gap-6">
        <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
          <Logo />
          <span>AfroViewz</span>
        </Link>
        <div className="ml-auto flex gap-1 text-sm">
          {links.map((l) => {
            const active = l.href === "/" ? path === "/" : path?.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`px-3 py-1.5 rounded-md whitespace-nowrap transition-colors ${
                  active ? "bg-amber text-amber-ink font-medium" : "text-muted hover:text-ink"
                }`}
              >
                <span className="hidden sm:inline">{l.label}</span>
                <span className="sm:hidden">{l.short ?? l.label}</span>
              </Link>
            );
          })}
        </div>
      </nav>
    </header>
  );
}
