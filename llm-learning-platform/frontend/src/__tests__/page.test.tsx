import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import Home from "../app/page";

// Mock next/link for testing environment
vi.mock("next/link", () => ({
    default: ({
        children,
        href,
        ...props
    }: {
        children: React.ReactNode;
        href: string;
        [key: string]: unknown;
    }) => (
        <a href={href} {...props}>
            {children}
        </a>
    ),
}));

// Mock lucide-react icons
vi.mock("lucide-react", () =>
    new Proxy(
        {},
        {
            get: (_, name) => {
                if (name === "__esModule") return true;
                return (props: Record<string, unknown>) => (
                    <svg data-testid={`icon-${String(name)}`} {...props} />
                );
            },
        },
    ),
);

describe("Home Page", () => {
    it("renders the main heading", () => {
        render(<Home />);
        const heading = screen.getByText(/Interactive LLM/i);
        expect(heading).toBeInTheDocument();
    });

    it("renders module cards", () => {
        render(<Home />);
        // Check for module categories
        const core = screen.getByText(/Core Foundations/i);
        expect(core).toBeInTheDocument();
    });

    it("renders module links with correct hrefs", () => {
        render(<Home />);
        const links = screen.getAllByRole("link");
        // Should have at least a few module links
        expect(links.length).toBeGreaterThan(0);
        // At least one should link to a module page
        const moduleLinks = links.filter(
            (link) =>
                link.getAttribute("href")?.includes("/modules/") ||
                link.getAttribute("href")?.includes("/tokenization"),
        );
        expect(moduleLinks.length).toBeGreaterThan(0);
    });

    it("displays module difficulty levels", () => {
        render(<Home />);
        // Check that difficulty indicators are present
        const beginnerBadges = screen.getAllByText(/Beginner/i);
        expect(beginnerBadges.length).toBeGreaterThan(0);
    });
});
