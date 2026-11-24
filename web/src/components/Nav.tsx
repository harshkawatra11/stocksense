import Link from "next/link";

export const NAV: { href: string; label: string }[] = [
  { href: "/picks/", label: "Daily Picks" },
  { href: "/#smart", label: "Smart Money" },
  { href: "/#sector", label: "Sector Map" },
  { href: "/#lab", label: "Swing Lab" },
  { href: "/flow/", label: "Money Flow" },
  { href: "/of/", label: "Order Flow" },
  { href: "/strategies/", label: "Strategies" },
  { href: "/learn/", label: "Learn" },
];

export function Nav() {
  return (
    <nav id="topnav" aria-label="Main">
      {NAV.map((n) => (
        <Link key={n.href} href={n.href}>
          {n.label}
        </Link>
      ))}
    </nav>
  );
}
